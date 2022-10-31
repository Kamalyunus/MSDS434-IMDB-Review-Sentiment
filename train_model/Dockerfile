FROM tensorflow/tensorflow:2.2.2-gpu
WORKDIR /root
COPY train_model/. ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "train_imdb.py"]