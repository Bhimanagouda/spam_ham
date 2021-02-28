import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv(r'spam.csv', encoding = 'latin-1')
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)
df = df.rename(columns={"v1":"target", "v2":"message"})
def preprocess(text):
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.strip()
    text = text.split(" ") 
    return text
	
def create_bow(docs):
    
    bow = []
    for d in docs:
        count = dict()
        for words in d:
            count[words] = count.get(words, 0) + 1
        bow.append(count)
        
    return bow

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.strip()
    text = text.split()
    text = ' '.join(list(filter(lambda x : x not in ['', ' '], text)))
    return text

df.message = df.message.apply(preprocess_text)	

X_train, X_test, y_train, y_test = train_test_split(df.message.values, df.target.values, test_size=0.1, stratify=df.target)

bow = CountVectorizer(stop_words='english')
bow.fit(X_train)

# Save the vectorizer
import pickle
vec_file = 'vectorizer.pkl'
pickle.dump(bow, open(vec_file, 'wb'))

X_train = bow.transform(X_train)
X_test = bow.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f'Accuracy : {accuracy_score(y_test, naive_bayes.predict(X_test)):.3f}')
print(f'Precision : {precision_score(y_test, naive_bayes.predict(X_test), pos_label="spam"):.3f}')
print(f'Recall : {recall_score(y_test, naive_bayes.predict(X_test), pos_label="spam"):.3f}')
print(f'F1-Score : {f1_score(y_test, naive_bayes.predict(X_test), pos_label="spam"):.3f}')

import pickle
pickle_out = open("spam_ham.pkl","wb")
pickle.dump(naive_bayes, pickle_out)
pickle_out.close()

