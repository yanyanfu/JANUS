FROM tensorflow/tensorflow:2.0.0rc0-gpu-py3

ENV PATH="/.local/bin:${PATH}"

ADD ./requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update -y && apt-get install git wget ffmpeg openjdk-11-jdk-headless maven -y

EXPOSE 8888 6006
