FROM python:3.10

# Set the working directory
WORKDIR /src/dp

# copy requirements file over
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the data processing file over
COPY DataProcessing.py ./

# Run the application
CMD ["python", "DataProcessing.py"]
