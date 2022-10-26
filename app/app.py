import streamlit as st
import requests
import json

predictionEndpoint = 'https://sentiment-nnlg3yqwxa-ue.a.run.app/sentiment'

st.set_page_config(
    page_title="IMDB Sentiment",
    page_icon="random",
    menu_items={
        'About': "# MSDS 434 Final Project: *IMDB Sentiment Analysis*"
    }
)

@st.cache(suppress_st_warning=True)
def getPrediction(text):
    data = {'text': text}
    response=requests.post(predictionEndpoint, json = data)
    responseTime=response.elapsed.total_seconds()*1000

    return round(responseTime,2), response

st.title('IMDB Reviews: Sentiment Analysis')
st.write('The model is trained on 50,000 movie reviews. It takes the movie review entered by user below and classify it as positive or negative review.')
st.markdown('The model training and prediction code is covered in the corresponding GitHub repository https://github.com/Kamalyunus/MSDS434-IMDB-Review-Sentiment')
text_input = st.text_area('Please Enter a Movie Review:')

st.write("The first request might take a while (GCP cloud run service scales to zero to save cost).")

if st.button("Predict"):
    responseTime, response = getPrediction(text_input)
    prediction = json.loads(response.text)

    st.write("Response Time (in ms): ",responseTime)
    st.write("Movie Review Model Prediction:", prediction["sentiment"].upper())
        
    st.bar_chart({"Model Prediction":prediction["confidence"]})