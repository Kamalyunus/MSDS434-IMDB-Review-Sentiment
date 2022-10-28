export JOB_NAME=imdb_$(date +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/trusty-bearing-366601/imdb-training:latest

export REGION=us-central1

gcloud config set project trusty-bearing-366601

gcloud ai-platform jobs submit training $JOB_NAME \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  --scale-tier=BASIC_GPU 