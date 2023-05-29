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
    input_text = pd.Series(request.form['text'])[0]
    input_text = preprocess_text(input_text)

    # Convert the text to sequence of words
    X_test_sequences = tokenizer.texts_to_sequences([input_text])

    # Pad the sequence
    X_test_padded = pad_sequences(X_test_sequences, maxlen=len(X_test_sequences), padding='post')

    # Make predictions
    predictions = model.predict(X_test_padded)

    # Get the sentiment with the highest probability
    sentiment = np.argmax(predictions)

    # Put name to the sentiment

    if sentiment == 0:
        sentiment = 'Neutral'
    elif sentiment == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    
    return render_template('tweets_form.html', prediction_text='The tweet has a {} sentiment'.format(sentiment))

if __name__ == "__main__":

    app.run(debug=True)




