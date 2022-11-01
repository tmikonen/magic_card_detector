FROM python:3.10.8-slim-buster as mtg-card-detector

# installing CV2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /src

RUN pip install --upgrade pip
COPY /requirements/card_detector_requirements.txt requirements.txt
RUN pip install -r requirements.txt

# keep alive
#CMD ["tail", "-f", "/dev/null"]
