FROM python:3.10-slim

# set a directory for the app
WORKDIR /src/mt

# copy requirements file over
COPY ../../requirements.txt ./

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the model training file over
COPY ModelTraining.py ./

# run the command
CMD ["python", "ModelTraining.py"]

