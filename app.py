import streamlit as st
import nltk
import joblib
nltk.download('stopwords')
from tensorflow import keras
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from tensorflow.keras.preprocessing import sequence

news_title = ""
model = keras.models.load_model('model.h5')

st.title("Fake News Detection")

st.header("CHECK YOUR NEWS HERE!")
news_title = st.text_area('Enter your news title below')

X = list()
X.append(news_title)
tokenizer = open('tokenizer.pkl', 'rb')
tokenized = joblib.load(tokenizer)
max_len = 150
tokenized_pred = tokenized.texts_to_sequences(X)
X = sequence.pad_sequences(tokenized_pred, maxlen=max_len)

prediction = model.predict_classes(X)

if st.button("Detect"):
        if prediction[0] == 1:
          st.success("Your news is FAKE!")
        else:
            st.success("Your news is REAL!")
