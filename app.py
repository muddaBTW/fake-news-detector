import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Load model & vectorizer ---
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# --- NLP tools ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Preprocessing function (same as training) for user data ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)

    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# --- Streamlit UI ---
st.title("Fake News Detector")
st.write("Enter a news article to check whether it is Fake or Real.")

user_input = st.text_area("Paste news article here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        cleaned = preprocess(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0] # 0 to extract the value 0 or 1

        if prediction == 1:
            st.error("🚨 This looks like FAKE news.")
        else:
            st.success("✅ This looks like REAL news.")