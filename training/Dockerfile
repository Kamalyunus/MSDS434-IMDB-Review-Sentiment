FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /root
COPY training/. ./
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "train.py"]