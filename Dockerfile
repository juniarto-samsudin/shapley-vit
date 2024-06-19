# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app 

# Create /app/logs 
RUN mkdir -p /app/logs /app/local1 /app/local2 /app/local3 /app/global /app/valdataset

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install opencv-python==4.9.0.80 \
    && pip3 install scikit-learn==1.4.2 \
    && pip3 install pandas==2.0.3 \
    && pip3 install matplotlib==3.8.4 \
    && pip3 install tqdm==4.66.2 \
    && pip3 install prettytable \
    && pip3 install transformers==4.41.2 \
    && pip3 install python-dotenv==1.0.1 \
    && pip3 install peft==0.11.1 \
    && pip3 install redis==5.0.6   

CMD ["python", "mainShapley.py"]
