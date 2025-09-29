import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# --- Configuration ---
MODEL_PATH = 'bilstm_sentiment_model.h5'
INDEX_PATH = 'word_index.json'
MAXLEN = 200  # Must match the padding length used during training

# --- 1. Load Model and Assets ---
@st.cache_resource  # Caches the model so it only loads once
def load_assets():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(INDEX_PATH, 'r') as f:
            word_index = json.load(f)
        return model, word_index
    except FileNotFoundError:
        st.error(f"Error: Model file ({MODEL_PATH}) or index file ({INDEX_PATH}) not found.")
        st.stop()

model, word_index = load_assets()

# Reverse the word index for tokenization function
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Add special tokens offset (IMDb uses 1 for start, 2 for unknown, 3 for skip)
word_to_id = {k:(v+3) for k,v in word_index.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2


# --- 2. Prediction Function ---
def predict_sentiment(review_text):
    # Tokenize the review
    tokens = [word_to_id.get(word.lower(), 2) for word in review_text.split()]
    
    # Pad the sequence
    padded_sequence = pad_sequences([tokens], maxlen=MAXLEN)
    
    # Make the prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    return prediction

# --- 3. Streamlit Interface ---
st.title("🍿 Bi-LSTM Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below to see if our Deep Learning model thinks it's **Positive** or **Negative**.")

# User input
review_input = st.text_area("Your Movie Review:", "This movie was absolutely brilliant, the acting was superb and the plot twist was unexpected!")

if st.button("Analyze Sentiment"):
    if review_input:
        with st.spinner('Analyzing...'):
            score = predict_sentiment(review_input)
            
            # Determine sentiment
            if score > 0.5:
                sentiment = "Positive"
                st.success(f"Prediction: **{sentiment}** (Confidence: {score:.2f}) 👍")
            else:
                sentiment = "Negative"
                st.error(f"Prediction: **{sentiment}** (Confidence: {1 - score:.2f}) 👎")
    else:
        st.warning("Please enter a review to analyze.")