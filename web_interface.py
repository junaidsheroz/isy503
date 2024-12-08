# pip install streamlit
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Title
st.title("ISY503 Assessment 3 Groupf")

# Text area
new_reviews = st.text_input("Enter Reviews to be Analyzed")
new_reviews = list(new_reviews)


# Button
if st.button("Analyze"):
    model = load_model("sentiment_analyzer.h5")
    new_reviews = list(new_reviews)
    tokenizer = Tokenizer(num_words=10000)
    new_sequences = tokenizer.texts_to_sequences(new_reviews)
    new_padded = pad_sequences(new_sequences, maxlen=100, padding="post")
    # Predict the sentiment
    prediction = model.predict(new_padded)
    # Display the result
    if prediction[0] > 0.5:
        st.success("Positive review")
    else:
        st.error("Negative review")
