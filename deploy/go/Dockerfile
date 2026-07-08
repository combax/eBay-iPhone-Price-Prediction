# syntax=docker/dockerfile:1
# Go + ONNX serving image for Kubernetes.
#
# Build from the REPO ROOT (needs artifacts/serving/ from `python -m ebay_price.export`):
#   docker build -f deploy/go/Dockerfile -t ebay-price-go .
ARG ORT_VERSION=1.22.0

FROM golang:1.23-bookworm AS build
WORKDIR /src
COPY deploy/go/go.mod deploy/go/go.sum ./
RUN go mod download
COPY deploy/go/*.go ./
RUN CGO_ENABLED=1 go build -o /server .

FROM debian:bookworm-slim
ARG ORT_VERSION
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ADD https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz /tmp/ort.tgz
RUN tar -xzf /tmp/ort.tgz -C /tmp \
    && cp -a /tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* /usr/local/lib/ \
    && rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-x64-${ORT_VERSION} \
    && useradd -r server
COPY --from=build /server /usr/local/bin/server
COPY artifacts/serving /app/serving
ENV SERVING_DIR=/app/serving \
    ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so
USER server
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/server"]
