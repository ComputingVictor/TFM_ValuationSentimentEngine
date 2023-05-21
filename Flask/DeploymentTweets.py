# Import libraries.

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
import pickle


def preprocess_text(text):

    '''
    Preprocesses the text to be used in the model.

    '''

    stop_words = set(stopwords.words('english'))

    text = text.str.lower()
    text = text.str.replace('@[^\s]+','')
    text = text.str.replace('http\S+|www.\S+','')
    text = text.str.replace('[\:\;\=][\-\^]?[\(\)\[\]\{\}\@D\|Pp\$\*\+\#]','')
    text = text.str.replace('[^\w\s]','')
    text = text.str.replace('\n',' ')
    text = text.apply(nltk.word_tokenize)
    text = text.apply(lambda x: [item for item in x if item not in stop_words])
    text_processed = text.apply(lambda x: ' '.join(x))

    return text_processed

# Load the model.

model = tf.keras.models.load_model('../models/nn_tweets.h5', compile=False)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the tokenizer that trained the Neuronal Network.

with open('../models/tweets_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Flask app.

app = Flask(__name__)

@app.route('/')

def index():

    '''
    Renders the home page.
    '''

    return render_template('tweets_form.html')

@app.route('/predict', methods=['POST'])

def predict():
    '''
    Predicts the class of the review.
    '''
    input_text = pd.Series(request.form['text'])
    input_text = preprocess_text(input_text)
    input_text = tokenizer.texts_to_sequences(input_text)
    input_text = pad_sequences(input_text, maxlen=25)
    prediction = model.predict(input_text)[0]
    class_predicted = np.argmax(prediction) + 1
    clases_texto = {
    1: "Neutral",
    2: "Positive",
    3: "Negative"
    }

    output = clases_texto[class_predicted]
    
    return render_template('tweets_form.html', prediction_text='The tweet has a {} sentiment'.format(output))

if __name__ == "__main__":

    app.run(debug=True)




