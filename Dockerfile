FROM ubuntu


RUN apt-get update -y && apt-get install -y python-pip python-dev curl
RUN pip install --upgrade pip
RUN apt-get install python2-dev
RUN apt-get install python-dev-is-python3

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt


HEALTHCHECK --interval=10s --timeout=30s CMD curl --fail http://localhost:5000/ || exit 1

ENTRYPOINT ["python"]
CMD ["app.py"]