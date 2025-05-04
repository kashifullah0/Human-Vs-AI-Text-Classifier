import streamlit as st
import joblib

class AIHUMAN:
    
    def __init__(self):
        self.we = joblib.load("wv.pkl")
        self.model = joblib.load("model.pkl")
        self.UI = self.user_interface()

    def user_interface(self):
        st.title("AI VS Human Text Classfier")
        st.text_area("Enter your Text")



if __name__ == "__main__":
    app = AIHUMAN()