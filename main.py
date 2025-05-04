import streamlit as st
import joblib
import string
import nltk

# Ensure necessary resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

class AIHUMAN:
    
    def __init__(self):
        self.vectorizer = joblib.load("wv.pkl")
        self.model = joblib.load("model.pkl")
        self.run_app()

    def preprocess_text(self, text):
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        tokens = word_tokenize(text.lower())
        filtered_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
        return " ".join(filtered_tokens)

    def run_app(self):
        st.title("AI vs Human Text Classifier")
        user_input = st.text_area("Enter your text:")

        if st.button("Predict"):
            processed_text = self.preprocess_text(user_input)
            vectorized_text = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(vectorized_text)

            if prediction[0] == 0:
                st.warning("The text is likely *AI-generated*.")
            else:
                st.success("The text is likely *Human-written*.")

if __name__ == "__main__":
    AIHUMAN()
