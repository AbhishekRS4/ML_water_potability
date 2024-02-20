# Deployment with kubernetes (locally)

## Deploy the docker image for the FastAPI ML application with kubernetes
* Install [kubectl](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html) and [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
* Setup a kind cluster with the following command
```
kind create cluster
```
* To check the cluster info, run the following commands
```
kubectl cluster-info --context kind-kind
```
* To check services, pods, deployments; run the following commands
```
kubectl get service
kubectl get pod
kubectl get deployment
```
* Load the docker image for the FastAPI ML application to the kubernetes cluster
```
kind load docker-image fastapi_water_potability
```
* Setup the deployment, the config for deployment is available in [deployment.yaml](deployment.yaml)
```
kubectl apply -f kubernetes_deployment/deployment.yaml
```
* Run port forwarding and check the deployment
```
kubectl get pod
kubectl port-forward <pod_name> 5000:5000
python3 test_post_request.py
```
* Setup the service, the config for service is available in [service.yaml](service.yaml)
```
kubectl apply -f kubernetes_deployment/service.yaml
```
* Run port forwarding and check the service
```
kubectl port-forward service/fastapi-water-potability 5000:80
python3 test_post_request.py
```
* **NOTE**: In the port forwarding command, the first port is port on the host, the second port
is the port of the deployment pod or the service which can be found in the respective yaml files
