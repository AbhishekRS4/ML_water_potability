# ML application for water potability classification


## Dataset Info
* The source of the dataset is the following [repo](https://github.com/MainakRepositor/Datasets/tree/master)
* The task is a binary classification task to predict the water potability given the different feature measurements of the water quality
* The dataset sample contains about 3.2K samples


## Project Info
* This repo contains a water potability Machine Learning FastAPI application deployment
* For the MLOps - MLFlow, FastAPI, Docker, Kubernetes has been utilized
* For deployment, an ML backend API app has been developed and deployed using FastAPI, docker, kubernetes
* For the training, the dataset is split into 90% - 10% for train and test sets respectively
* For getting the latest model from the mlflow logs for production, use the script [src/get_model_for_production.py](src/get_model_for_production.py)
* The python packages are listed in [requirements.txt](requirements.txt)
* The docker container can be deployed using [src/backend_app/Dockerfile](src/backend_app/Dockerfile)
* For training and logging the model, use the [src/modeling/ml_model_dev.py](src/modeling/ml_model_dev.py) script
* The FastAPI app deployment code is in [src/backend_app/app.py](src/backend_app/app.py) script
* The Kubernetes deployment configuration files can be found in [src/deploy/](src/deploy/)
* To test the deployed FastAPI app on a local machine, the [src/test_post_request.py](src/test_post_request.py) script can be used
* This project involved learning orchestrating the scaling of the deployed ML service app with kubernetes


## Docker deployment of ML backend app on a local machine
* To build the container for the backend app, run the following command from the repo root
```
docker build -f src/backend_app/Dockerfile -t ml-water-potability-backend .
```
* Use `--no-cache` in the above command to build the image without using cached layers
* To the run the container with the backend app, run the following command
```
docker run -p 5000:5000 -t ml-water-potability-backend
```


## Kubernetes deployment of ML backend app on a local machine
* The deployment architecture used in this project is Serverful (Raw deployment)
* To deploy the container on a kubernetes cluster, refer [src/deploy/README.md](src/deploy/README.md) for detailed instructions
* Alternatively, run [src/deploy/start_deployment.sh](src/deploy/start_deployment.sh) for starting the kubernetes deployment


## HuggingFace deployment
* The FastAPI application with appropriate changes has also been deployed to [HuggingFace](https://huggingface.co/spaces/abhishekrs4/ML_water_potability)
* To test the deployed FastAPI app on HuggingFace, use the [src/test_post_request.py](https://huggingface.co/spaces/abhishekrs4/ML_water_potability/blob/main/test_post_request.py) script in the HuggingFace repo since the endpoint is different


## Documentation
The documentation generated with sphinx is available in [docs/_build/html/index.html](docs/_build/html/index.html)
