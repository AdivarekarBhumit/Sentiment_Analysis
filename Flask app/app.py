from flask import render_template, request, Flask
import helpers.utils as utils 
from keras.models import model_from_json
import pickle as pb   
import numpy as np
from keras.preprocessing.sequence import pad_sequences 
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras import backend as k 
import os
import glob

with open('./helpers/helper.pb', 'rb') as f:
    args = pb.load(f)
vocab_to_int = args['vocab_to_int']
int_to_vocab = args['int_to_vocab']
maxlen = args['maxlen']

def get_model():
    with open('./helpers/sentiment.json', 'r') as f:
        load_model = f.read()
    model = model_from_json(load_model)
    model.load_weights('./helpers/sentiment100cells.h5')
    return model

def predict_sentiment(sequences):
    model = get_model()
    preds = model.predict(sequences)
    k.clear_session()
    return preds[0]

def text_to_seq(text, max_len):
    text = utils.clean_text(text)
    seq = []
    for word in text.split():
        seq.append(vocab_to_int[word])
    pad_seq = pad_sequences([seq], maxlen=max_len)
    return pad_seq

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if os.path.exists('./static/'):
        pass
    else:
        os.mkdir('./static/')
    for i in glob.glob('./static/*.jpg'):
        os.remove(i)
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    og_comment = request.form.get('comment')
    sequences = text_to_seq(og_comment, maxlen)
    preds = predict_sentiment(sequences)
    neg_sentiment = preds[0]
    pos_sentiment = preds[1]
    labels = ['Positive', 'Negative']
    sizes = [pos_sentiment * 100, neg_sentiment * 100]
    colors = ['orange', 'yellow']
    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie(sizes, colors=colors, labels=labels,
                                         autopct='%1.2f%%', startangle=90)
    path = './static/sentiment' + str(np.random.randint(1,1000)) + '.jpg'

    for text in texts:
        text.set_color('black')

    for autotext in autotexts:
        autotext.set_color('black')
    
    ax1.axis('equal')
    plt.tight_layout()
    plt.plot()
    plt.savefig(path)
    return render_template('analyze.html', og_comment=og_comment, path=path)

if __name__ == '__main__':
    app.run(debug=True)