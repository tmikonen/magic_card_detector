FROM python:3.10.8-slim-buster

RUN apt-get update
RUN apt-get install python3-psycopg2 -y

WORKDIR /src

RUN pip install --upgrade pip
COPY /requirements/inventory_requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
## keep alive
#CMD ["tail", "-f", "/dev/null"]