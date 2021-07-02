FROM pytorchlightning/pytorch_lightning:1.3.8-py3.8-torch1.8

COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ENV HOME=/usr/src/app