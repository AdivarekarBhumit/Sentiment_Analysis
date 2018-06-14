import pandas as pd  
import numpy as np 
import tensorflow as tf  
import os  
import utils 

data_df = utils.open_csv()
labels = data_df[1]

all_text = data_df[0].values.tolist()
new_text = []

## Clean text
for text in all_text:
    new_text.append(utils.clean_text(text))

wc = utils.word_count(cleaned_text_list=new_text)

embedding_index = utils.create_embeddings_of_word2vec()

vocab_to_int, int_to_vocab = utils.vocab_to_int(wc, embedding_index)

word_embedding_matrix = utils.final_embedding_matrix(vocab_to_int, embedding_index)

