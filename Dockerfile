FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /root

COPY requirements.txt ./requirements.txt
COPY train.py ./train.py

RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "train.py"]