import streamlit as st
import joblib
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer



class AIHUMAN:
    
    def __init__(self):
        self.we = joblib.load("wv.pkl")
        self.model = joblib.load("model.pkl")
        self.UI = self.user_interface()

    def user_interface(self):
        stop_words = set(stopwords.words("english"))
        punctuations = set(string.punctuation)
        ps = PorterStemmer()
        st.title("AI VS Human Text Classfier")
        text = st.text_area("Enter your Text")
        tokens = nltk.word_tokenize(text.lower())
        
        filtered_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
        text = " ".join(filtered_tokens)
        word_to_vec = self.we.transform([text])
        prediction = self.model.predict(word_to_vec)
        if st.button("Prediction"):
            word_to_vec = self.we.transform([text])
            prediction = self.model.predict(word_to_vec)
            if prediction[0] == 0:
                st.warning("AI Generted")
            else:
                st.success("Human Generted")



if __name__ == "__main__":
    app = AIHUMAN()