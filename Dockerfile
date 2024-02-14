FROM python:3.8.10-slim

WORKDIR /app

# install linux package dependencies
RUN apt-get update -y
RUN apt-get install -y libgomp1

# can copy files only from current working directory where docker builds
# cannot copy files from arbitrary directories

COPY ./model_for_production/ /data/model_for_production
COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./modeling/*.py ./modeling/
COPY ./*.py .

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
