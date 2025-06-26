FROM 3.11-bullseye

# Upgrade pip and install system updates to reduce vulnerabilities
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends \
	build-essential gcc libglib2.0-0 libsm6 libxext6 libxrender-dev \
	&& apt-get clean && rm -rf /var/lib/apt/lists/* \
	&& pip install --upgrade pip

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt 

WORKDIR /data

ENTRYPOINT ["python", "/app/neural_style/StyleMotion_adain.ipynb"]