import pandas as pd
import numpy as np
import re
import string
import joblib
import streamlit as st

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def output_label(n):
    return "Fake News" if n == 0 else "Real News"

def main():
    st.title("Fake News Detection App")
    st.write("Enter news text below to check if it's real or fake.")
    
    user_input = st.text_area("Enter news content:")
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            input_text = [wordopt(user_input)]
            input_vector = vectorizer.transform(input_text)
            prediction = model.predict(input_vector)
            
            st.subheader("Prediction:")
            st.write(f"{output_label(prediction[0])}")

if __name__ == "__main__":
    main()
