FROM python:3.9
WORKDIR /code
COPY . .

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install zip build-essential ffmpeg libsm6 libxext6 wget

RUN pip install -r requirements.txt
RUN pip install -r requirements-extras.txt