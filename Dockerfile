FROM ubuntu

RUN apt-get update -y && apt-get install -y python-pip python-dev-is-python3 curl

WORKDIR /app
COPY . /app
COPY *.py /pages/

# The commands below can be executed in any order
COPY  requirements.txt .

RUN apt-get update && apt-get install -y python3 python3-pip sudo
RUN pip3 install -r requirements.txt



HEALTHCHECK --interval=10s --timeout=30s CMD curl --fail http://localhost:5000/ || exit 1

EXPOSE 3000

CMD ["streamlit", "run", "app.py"]
