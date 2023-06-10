# Import libraries.

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):

    '''
    Preprocesses the text to be used in the model.

    '''


    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = text.replace('@[^\s]+','')
    text = text.replace('http\S+|www.\S+','')
    text = text.replace('[\:\;\=][\-\^]?[\(\)\[\]\{\}\@D\|Pp\$\*\+\#]','')
    text = text.replace('[^\w\s]','')
    text = text.replace('\n',' ')
    text = nltk.word_tokenize(text)
    text = [item for item in text if item not in stop_words]
    text_processed = ' '.join(text)

    return text_processed


# Load the ML model pkl.

with open('../models/logistic_model_resampled.pkl', 'rb') as handle:
    model = pickle.load(handle)


# Load the tokenizer pkl.

with open('../models/tfidf_vectorizer.pkl', 'rb') as handle:
    tfidf = pickle.load(handle)

# Flask app.

app = Flask(__name__)

@app.route('/')

def index():

    '''
    Renders the home page.
    '''

    return render_template('form.html')

@app.route('/predict', methods=['POST'])

def predict():
    '''
    Predicts the class of the review.
    '''
  
    input_text = pd.Series(request.form['text'])[0]
    input_text = preprocess_text(input_text)

    # Convert input_text to string.

    input_text = [input_text]

    # Transform the text.

    text_transformed = tfidf.transform(input_text)

    # Make predictions

    predictions = model.predict(text_transformed)

    # Get the sentiment with the highest probability


        
    return render_template('form.html', prediction_text='The review is {}'.format(' '.join(str(p) for p in predictions)))


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == "__main__":

    app.run(debug=True)




