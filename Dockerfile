FROM python:3.10-bullseye

WORKDIR /opt/app

ADD requirements.txt /opt/app/requirements.txt

RUN apt-get update -y \
&& apt-get upgrade -y \
&& python3 -m pip install -r requirements.txt

ADD process.py /opt/app/process.py

ENTRYPOINT ["/opt/app/process.py"]