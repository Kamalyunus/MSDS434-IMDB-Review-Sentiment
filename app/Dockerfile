FROM python:3.9-slim

WORKDIR /app
EXPOSE 8501

COPY app/ ./
RUN pip3 install streamlit google-cloud-aiplatform google-cloud pandas

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]