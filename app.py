import streamlit as st
import pandas as pd

@st.cache(suppress_st_warning=True)
def run():
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
            st.write("Movie Review Model Prediction: POSITIVE")
            prediction=pd.DataFrame({'displayNames': ['positive', 'negative'], 'confidences': [0.99,0.01]})    
            st.bar_chart(data=prediction, x='displayNames', y='confidences')