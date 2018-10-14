import pandas as pd  
import numpy as np 
import tensorflow as tf  
import os  
import utils 
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import pickle as pb    

embed_size = 300

data_df = utils.open_csv()
labels = data_df[1].values.tolist()

all_text = data_df[0].values.tolist()
new_text = []

## Clean text
for text in all_text:
    new_text.append(utils.clean_text(text))

wc = utils.word_count(cleaned_text_list=new_text)

embedding_index = utils.create_embeddings_of_word2vec()

vocab_to_int, int_to_vocab = utils.vocab_to_int(wc, embedding_index)

word_embedding_matrix = utils.final_embedding_matrix(vocab_to_int, embedding_index)

## Change sentences to vocab_to_int representation
num_sentences = []
for text in new_text:
    num_sentences.append([vocab_to_int[word] for word in text.split()])

max_len = 0
for seq in num_sentences:
    max_len = max(max_len, len(seq))

args_dict = {
    'vocab_to_int':vocab_to_int,
    'int_to_vocab':int_to_vocab,
    'maxlen':max_len
}

with open('helper.pb', 'wb') as f:
    pb.dump(args_dict, f)

data = pad_sequences(num_sentences, maxlen=max_len)

labels = np.array(labels)
labels = to_categorical(labels, 2)
data = np.array(data)

max_features = 2040
def get_model():
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size, weights=[word_embedding_matrix])(inp)
    x = SpatialDropout1D(0.75)(x)
    x = Bidirectional(GRU(100, return_sequences=True,activation='relu', dropout=0.3, recurrent_dropout=0.))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(2, activation="softmax")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model()
print(model.summary())

## Splitting into training and validation set
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, train_size=0.90, random_state=255)

## Tensorboard Callback

tb = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True)
model.fit(X_train, Y_train, batch_size=32, epochs=15, validation_data=(X_val, Y_val), callbacks=[tb])

# model_json = model.to_json()
# with open('sentiment.json', 'w') as f:
#     f.write(model_json)

# ## Save model
# model.save_weights('sentimentnewcells.h5')

model.save('sentiment.hdf5')