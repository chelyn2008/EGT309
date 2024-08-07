FROM python:3.10-slim

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY ../../requirements.txt ./

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ModelTraining.py ./

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "ModelTraining.py"]

