FROM python:3.10

# set a directory for the app
WORKDIR /usr/src/mt

# copy requirements file over
COPY requirements.txt ./

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the model training file over
COPY ModelTraining.py ./

# run the command
CMD ["python", "ModelTraining.py"]

