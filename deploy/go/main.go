// eBay iPhone price serving for Kubernetes: a small Go server that rebuilds
// the sklearn feature vector from serving.json (constants exported from the
// fitted pipelines by `python -m ebay_price.export`) and runs the ONNX models.
//
// At startup it replays the recorded parity vectors and refuses to serve if
// its predictions drift from the python pipeline's.
//
// Endpoints match api.py: POST /predict, GET /health, GET /.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

// ---- serving spec (generated, never hand-edited) ----

type Prep struct {
	Numeric      []string            `json:"numeric"`
	Categorical  []string            `json:"categorical"`
	ImputeMedian []float64           `json:"impute_median"`
	ScaleMean    []float64           `json:"scale_mean"`
	ScaleStd     []float64           `json:"scale_std"`
	Categories   map[string][]string `json:"categories"`
}

type ParityCase struct {
	Input    map[string]any `json:"input"`
	Expected float64        `json:"expected"`
}

type Target struct {
	Onnx           string             `json:"onnx"`
	Prep           Prep               `json:"prep"`
	BandLogOffsets map[string]float64 `json:"band_log_offsets"`
	Parity         []ParityCase       `json:"parity"`

	session *ort.DynamicAdvancedSession
}

type Spec struct {
	TrainedOn string             `json:"trained_on"`
	Vocab     map[string][]any   `json:"vocab"`
	Targets   map[string]*Target `json:"targets"`
}

// vector mirrors the sklearn ColumnTransformer: scaled numerics then one-hot
// categoricals, in spec order. NaN numerics take the training median and
// unknown categories one-hot to all zeros (handle_unknown="ignore").
func (p *Prep) vector(row map[string]any) []float32 {
	n := len(p.Numeric)
	for _, cats := range p.Categories {
		n += len(cats)
	}
	out := make([]float32, 0, n)
	for i, col := range p.Numeric {
		v, ok := row[col].(float64)
		if !ok || math.IsNaN(v) {
			v = p.ImputeMedian[i]
		}
		out = append(out, float32((v-p.ScaleMean[i])/p.ScaleStd[i]))
	}
	for _, col := range p.Categorical {
		s, _ := row[col].(string)
		for _, c := range p.Categories[col] {
			if c == s {
				out = append(out, 1)
			} else {
				out = append(out, 0)
			}
		}
	}
	return out
}

// predict returns the raw-scale point prediction (the ONNX graphs emit
// log1p-scale values, exactly like the wrapped sklearn regressors).
func (t *Target) predict(row map[string]any) (float64, error) {
	vec := t.Prep.vector(row)
	input, err := ort.NewTensor(ort.NewShape(1, int64(len(vec))), vec)
	if err != nil {
		return 0, err
	}
	defer input.Destroy()
	outputs := []ort.Value{nil}
	if err := t.session.Run([]ort.Value{input}, outputs); err != nil {
		return 0, err
	}
	out := outputs[0].(*ort.Tensor[float32])
	defer out.Destroy()
	logPred := float64(out.GetData()[0])
	return math.Max(math.Expm1(logPred), 0), nil
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }

func (t *Target) band(pred float64) [2]float64 {
	lp := math.Log1p(pred)
	return [2]float64{
		round2(math.Max(math.Expm1(lp+t.BandLogOffsets["p10"]), 0)),
		round2(math.Expm1(lp + t.BandLogOffsets["p90"])),
	}
}

// ---- request handling (contract mirrors api.py) ----

type Listing struct {
	Condition           string   `json:"condition"`
	Model               string   `json:"model"`
	StorageGB           *float64 `json:"storage_gb"`
	CarrierStatus       string   `json:"carrier_status"`
	Location            *string  `json:"location"`
	SellerFeedbackPct   *float64 `json:"seller_feedback_pct"`
	SellerFeedbackCount *float64 `json:"seller_feedback_count"`
	ProductStars        *float64 `json:"product_stars"`
	ProductRatingsCount *float64 `json:"product_ratings_count"`
	Sealed              *bool    `json:"sealed"`
	BatteryHealthPct    *float64 `json:"battery_health_pct"`
}

type Prediction struct {
	PredictedPriceCad    *float64    `json:"predicted_price_cad"`
	PriceRangeCad        *[2]float64 `json:"price_range_cad"`
	PredictedShippingCad *float64    `json:"predicted_shipping_cad"`
	ShippingRangeCad     *[2]float64 `json:"shipping_range_cad"`
	TrainedOn            string      `json:"trained_on"`
}

func (s *Spec) allowed(field string, value any) bool {
	for _, v := range s.Vocab[field] {
		if v == value {
			return true
		}
	}
	return false
}

func (s *Spec) buildRow(l Listing) (map[string]any, error) {
	if l.StorageGB == nil {
		return nil, fmt.Errorf("storage_gb is required")
	}
	for field, value := range map[string]any{
		"condition": l.Condition, "model": l.Model,
		"carrier_status": l.CarrierStatus, "storage_gb": *l.StorageGB,
	} {
		if !s.allowed(field, value) {
			return nil, fmt.Errorf("%s %v is not one of %v", field, value, s.Vocab[field])
		}
	}
	// omitted numerics stay NaN and take the training medians in Prep.vector,
	// mirroring the python pipelines' imputers
	nan := math.NaN()
	row := map[string]any{
		"condition": l.Condition, "model": l.Model, "carrier_status": l.CarrierStatus,
		"location": "Canada", "storage_gb": *l.StorageGB,
		"seller_feedback_pct": nan, "seller_feedback_count": nan,
		"product_stars": nan, "product_ratings_count": 0.0,
		"sealed": 0.0, "battery_health_pct": nan,
	}
	if l.Location != nil {
		row["location"] = *l.Location
	}
	if l.SellerFeedbackPct != nil {
		row["seller_feedback_pct"] = *l.SellerFeedbackPct
	}
	if l.SellerFeedbackCount != nil {
		row["seller_feedback_count"] = *l.SellerFeedbackCount
	}
	if l.ProductStars != nil {
		row["product_stars"] = *l.ProductStars
	}
	if l.ProductRatingsCount != nil {
		row["product_ratings_count"] = *l.ProductRatingsCount
	}
	if l.Sealed != nil && *l.Sealed {
		row["sealed"] = 1.0
	}
	if l.BatteryHealthPct != nil {
		row["battery_health_pct"] = *l.BatteryHealthPct
	}
	return row, nil
}

func fail(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"detail": msg}) //nolint:errcheck
}

func (s *Spec) predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "POST only")
		return
	}
	var listing Listing
	if err := json.NewDecoder(r.Body).Decode(&listing); err != nil {
		fail(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	row, err := s.buildRow(listing)
	if err != nil {
		fail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	resp := Prediction{TrainedOn: s.TrainedOn}
	for name, t := range s.Targets {
		pred, err := t.predict(row)
		if err != nil {
			fail(w, http.StatusInternalServerError, name+": "+err.Error())
			return
		}
		point, rng := round2(pred), t.band(round2(pred))
		switch name {
		case "price":
			resp.PredictedPriceCad, resp.PriceRangeCad = &point, &rng
		case "shipping":
			resp.PredictedShippingCad, resp.ShippingRangeCad = &point, &rng
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp) //nolint:errcheck
}

// parityCheck replays the exported vectors through this server's own
// preprocessing + ONNX path and compares against the joblib pipeline's
// predictions recorded at export time.
func (s *Spec) parityCheck() error {
	for name, t := range s.Targets {
		for i, c := range t.Parity {
			row := make(map[string]any, len(c.Input))
			for k, v := range c.Input {
				if v == nil {
					row[k] = math.NaN()
				} else {
					row[k] = v
				}
			}
			got, err := t.predict(row)
			if err != nil {
				return fmt.Errorf("%s parity %d: %w", name, i, err)
			}
			if rel := math.Abs(got-c.Expected) / math.Max(c.Expected, 1); rel > 0.005 {
				return fmt.Errorf("%s parity %d: got %.4f want %.4f (drift %.2f%%)",
					name, i, got, c.Expected, 100*rel)
			}
		}
		log.Printf("%s: %d parity vectors passed", name, len(t.Parity))
	}
	return nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	servingDir := env("SERVING_DIR", "artifacts/serving")
	raw, err := os.ReadFile(filepath.Join(servingDir, "serving.json"))
	if err != nil {
		log.Fatalf("read serving spec: %v (run `python -m ebay_price.export` first)", err)
	}
	var spec Spec
	if err := json.Unmarshal(raw, &spec); err != nil {
		log.Fatalf("parse serving.json: %v", err)
	}

	ort.SetSharedLibraryPath(env("ONNXRUNTIME_LIB", "/usr/local/lib/libonnxruntime.so"))
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	for name, t := range spec.Targets {
		path := filepath.Join(servingDir, t.Onnx)
		inputs, outputs, err := ort.GetInputOutputInfo(path)
		if err != nil || len(inputs) != 1 || len(outputs) < 1 {
			log.Fatalf("%s: inspect %s: %v", name, path, err)
		}
		t.session, err = ort.NewDynamicAdvancedSession(
			path, []string{inputs[0].Name}, []string{outputs[0].Name}, nil)
		if err != nil {
			log.Fatalf("%s: load %s: %v", name, path, err)
		}
	}
	if err := spec.parityCheck(); err != nil {
		log.Fatalf("PARITY FAILURE — refusing to serve: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/predict", spec.predictHandler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"status": "ok", "parity": "passed", "trained_on": spec.TrainedOn,
		})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"service": "ebay-price-go", "endpoints": []string{"/predict", "/health"},
		})
	})

	addr := ":" + env("PORT", "8080")
	log.Printf("serving on %s (trained_on=%s)", addr, spec.TrainedOn)
	log.Fatal(http.ListenAndServe(addr, mux))
}
