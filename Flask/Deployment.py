# Import libraries.

import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify, render_template
import pickle


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))

    text = text.str.lower()
    text = text.str.replace('[\:\;\=][\-\^]?[\(\)\[\]\{\}\@D\|Pp\$\*\+\#]','')
    text = text.str.replace('[^\w\s]','')
    text = text.str.replace('\n',' ')
    text = text.apply(nltk.word_tokenize)
    text = text.apply(lambda x: [item for item in x if item not in stop_words])
    text_processed = text.apply(lambda x: ' '.join(x))

    return text_processed

# Load the model.

model = tf.keras.models.load_model('../models/nn_reviews.h5', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the tokenizer in models.

with open('../models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Flask app.

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])

# Function to predict the sentiment of the review.

def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_text = pd.Series(request.form['text'])
    input_text = preprocess_text(input_text)
    input_text = tokenizer.texts_to_sequences(input_text)
    input_text = pad_sequences(input_text, maxlen=300)
    prediction = model.predict(input_text)[0]
    class_predicted = np.argmax(prediction) + 1
    clases_texto = {
    1: "1 estrella",
    2: "2 estrellas",
    3: "3 estrellas",
    4: "4 estrellas",
    5: "5 estrellas"
    }

    output = clases_texto[class_predicted]
    
    return render_template('form.html', prediction_text='The review is {}'.format(output))

if __name__ == "__main__":

    app.run(debug=True)




