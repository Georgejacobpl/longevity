import streamlit as st
import pandas as pd
import sqlite3
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load your pre-trained model (save this as model.pkl)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Simple UI
st.set_page_config(page_title="Supplement Analyzer", page_icon="💊")
st.title("💊 Supplement Review Analyzer")
st.write("Analyze supplement video reviews in real-time!")

# Input section
st.subheader("🔍 Analyze Video Sentiment")
video_title = st.text_input("Enter video title:", "This pre-workout gives me amazing energy!")

if st.button("Analyze Sentiment"):
    if video_title:
        # Predict
        prediction = model.predict(vectorizer.transform([video_title]))[0]
        confidence = model.predict_proba(vectorizer.transform([video_title])).max()
        
        # Display result
        emoji = "😊" if prediction == "positive" else "😞" if prediction == "negative" else "😐"
        st.success(f"**Sentiment:** {emoji} {prediction.upper()} ({(confidence*100):.1f}% confident)")
        
        # Show sample database
        st.subheader("📊 Sample Video Database")
        sample_data = pd.DataFrame({
            'Video Title': [
                "Best pre-workout ever!",
                "This supplement is terrible",
                "My daily vitamin routine"
            ],
            'Sentiment': ['positive', 'negative', 'neutral'],
            'Supplement': ['Pre-workout', 'BCAAs', 'Multivitamin']
        })
        st.dataframe(sample_data)

# Demo section
st.subheader("🚀 Quick Demo")
demo_titles = [
    "This supplement changed my life!",
    "Waste of money, doesn't work",
    "Decent product for the price"
]

for title in demo_titles:
    if st.button(f"Test: {title}"):
        pred = model.predict(vectorizer.transform([title]))[0]
        st.info(f"**{title}** → **{pred}**")
