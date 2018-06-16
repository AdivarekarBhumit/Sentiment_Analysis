import keras 
from keras.models import model_from_json
from keras.preprocessing import text, sequence
import numpy as np 
import pickle as pb  
import utils

## load pickle file 
with open('helper.pb', 'rb') as f:
    args = pb.load(f)

vocab_to_int = args['vocab_to_int']
int_to_vocab = args['int_to_vocab']
max_len = args['maxlen']

json_model = open('sentiment.json', 'r')
load_model = json_model.read()
json_model.close()

model = model_from_json(load_model)
model.load_weights('sentiment100cells.h5')

print('Enter your text here \n')
text = utils.clean_text(input())

## Now convert the text to sequence
seq = []
for word in text.split():
    seq.append(vocab_to_int[word])

seq = sequence.pad_sequences([seq], maxlen=max_len)

prediction = model.predict(seq)

neg_sentiment = prediction[0][0]
pos_sentiment = prediction[0][1]

print('Positive Sentiment:{:.2f}%'.format(pos_sentiment * 100))
print('Negative Sentiment:{:.2f}%'.format(neg_sentiment * 100))
