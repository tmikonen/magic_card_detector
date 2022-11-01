FROM python:3.10.8-slim-buster as mtg-ai-card-detector

# installing CV2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /src

COPY /requirements/dev.txt requirements.txt

RUN pip install -r requirements.txt

# keep alive
CMD ["tail", "-f", "/dev/null"]
