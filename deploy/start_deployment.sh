#!/bin/bash
kind delete cluster --name kind
sleep 5
kind create cluster --name kind --config deploy/kind-config.yaml
sleep 5
kubectl cluster-info --context kind-kind
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s
kind load docker-image ml-water-potability-backend:latest
sleep 2
kubectl apply -f deploy/deployment.yaml
sleep 5
kubectl apply -f deploy/service.yaml
sleep 5
kubectl apply -f deploy/ingress.yaml
