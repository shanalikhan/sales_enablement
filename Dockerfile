FROM python:3.8-slim

RUN mkdir -p /home/AE

COPY . /home/AE

WORKDIR /home/AE

RUN apt-get update && apt-get install -y python3.8-dev && apt install -y python3-pip && apt install -y 
RUN apt-get update && pip install --upgrade pip
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y awscli
RUN apt-get update && apt-get install -y ffmpeg

RUN apt-get update && python3.8 -m pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

CMD ["streamlit", "run", "streamlit-sales-app.py"]