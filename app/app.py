import streamlit as st
import pandas as pa
from google.cloud import aiplatform
import google.auth
import predict_text_classification_single_label_sample as pd

mycredentials = google.auth.default()
aiplatform.init(credentials=mycredentials)

st.set_page_config(
    page_title="IMDB Sentiment",
    page_icon="random",
    menu_items={
        'About': "# MSDS 434 Final Project: *IMDB Sentiment Analysis*"
    }
)

st.title('IMDB Reviews: Sentiment Analysis')
st.write('The model is trained on 50,000 movie reviews. It takes the movie review entered by user below and classify it as positive or negative review.')
st.markdown('The model training and prediction code is covered in the corresponding GitHub repository https://github.com/Kamalyunus/MSDS434-IMDB-Review-Sentiment')
text_input = st.text_area('Please Enter a Movie Review:')

st.write("The first request might take a while (GCP cloud run service scales to zero to save cost).")

if st.button("Predict"):
    with st.spinner('Please wait...'):
        prediction=pa.DataFrame(
            pd.predict_text_classification_single_label_sample(
            project="609731156916",
            endpoint_id="6424741110211411968",
            location="us-central1",
            content=text_input
            )
        )
        st.write("Movie Review Model Prediction:", prediction["displayNames"][prediction.confidences.idxmax()].upper())
            
        st.bar_chart(data=pa.DataFrame(prediction), x='displayNames', y='confidences')