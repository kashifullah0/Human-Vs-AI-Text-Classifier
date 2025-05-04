import joblib
import numpy as np

# Load the TF-IDF vectorizer and model
tfidf = joblib.load("wv.pkl")
model = joblib.load("model.pkl")

# Transform input text - keep as sparse matrix
text_to_predict = ["my name is kashif"]
data = tfidf.transform(text_to_predict)

# Predict without converting to numpy array
pred = model.predict(data)
print(pred)