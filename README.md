# ML application for water potability classification

## Dataset Info
* The source of the dataset is the following [repo](https://github.com/MainakRepositor/Datasets/tree/master)
* The task is a binary classification task to predict the water potability given the different feature measurements of the water quality
* The dataset sample contains about 3.2K samples

## Repo Info
* This repo contains a water potability Machine Learning FastAPI application deployment
* For the ML application development, training, and management; MLFlow has been utilized
* For deployment, an API has been developed and deployed using FastAPI and docker
* For the training, the dataset is split into 90% - 10% for train and test sets respectively
* The python packages are listed in [requirements.txt](requirements.txt)
* The docker container can be deployed using [Dockerfile](Dockerfile)
* For training and logging the model, use the [modeling/ml_model_dev.py](modeling/ml_model_dev.py) script
* The FastAPI app deployment code is in [app.py](app.py) script
* To test the deployed FastAPI app on a local machine, the [test_post_request.py](test_post_request.py) script can be used

## Docker deployment instructions on a local machine
* To build the container, run the following command
```
docker build -t fastapi_water_potability .
```
* To the run the container, run the following command
```
docker run -p 5000:5000 -t fastapi_water_potability
```

## HuggingFace deployment
* The FastAPI application with appropriate changes has also been deployed to [HuggingFace](https://huggingface.co/spaces/abhishekrs4/ML_water_potability)
* To test the deployed FastAPI app on HuggingFace, use the [test_post_request.py](https://huggingface.co/spaces/abhishekrs4/ML_water_potability/blob/main/test_post_request.py) script in the HuggingFace repo since the endpoint is different
