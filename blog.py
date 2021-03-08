import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
bow = CountVectorizer(stop_words='english')
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.strip()
    text = text.split()
    text = ' '.join(list(filter(lambda x : x not in ['', ' '], text)))
    return text


app = Flask(__name__)

loaded_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
classifier=pickle.load(open("spam_ham.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')
	
@app.route('/predict_ham_spam',methods = ['GET', 'POST'])
def predict_ham_spam():
    if request.method == "POST":
        #posted_data = request.get_json()
        posted_data = request.form.values()
#         text = posted_data['input']
        text=" ".join(posted_data)
        p_text = preprocess_text(text)
        print(p_text)
        p_text = loaded_vectorizer.transform([p_text])
        print(p_text)
        prediction=classifier.predict(p_text)
        print(prediction)
    return render_template('index.html', prediction_text='Prediction is :  {}'.format(prediction))

if __name__ == "__main__":
    app.run(threaded=true,host='0.0.0.0',port=8081)