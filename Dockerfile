FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /root

RUN pip3 install -r requirements.txt
COPY train.py ./train.py

ENTRYPOINT ["python3", "train.py"]