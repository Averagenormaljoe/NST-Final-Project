FROM python:3.11

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt 

WORKDIR /data

ENTRYPOINT ["python", "/app/neural_style/StyleMotion_adain.ipynb"]