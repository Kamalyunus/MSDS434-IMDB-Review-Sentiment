FROM python:3.9-slim

EXPOSE 8080
COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py","--server.enableCORS=false", "--server.enableWebsocketCompression=false","--server.port=8080", "--server.address=0.0.0.0"]