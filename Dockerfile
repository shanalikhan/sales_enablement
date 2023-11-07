FROM python:3.8-slim

RUN mkdir -p /home/AE

COPY . /home/AE

WORKDIR /home/AE

RUN apt-get update && apt-get install -y python3.8-dev && apt install -y python3-pip && apt install -y curl apt-transport-https
RUN apt-get update && pip install --upgrade pip
RUN apt-get update && apt-get install -y git

RUN apt-get update && python3.8 -m pip install -r requirements.txt

CMD ["python3.8", ""]