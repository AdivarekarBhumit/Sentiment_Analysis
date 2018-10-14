import pandas as pd
import numpy as np 
import glob
import os 
import pickle as pb    
from nltk.corpus import stopwords 
import re 


stop_words = set(stopwords.words('english'))

contractions = { 
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
            }

## We are using fasttext word embeddings

## The embeddings are already stored in pickle file
def open_csv(path='./Data/yelp_dataset.txt'):
    df = pd.read_csv(path, sep='\t', header=None)
    return df 

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = text.split()

    new_text = []

    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    
    text = ' '.join(new_text)

    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = ' '.join(text.split())

    return text

def word_count(cleaned_text_list):
    wc = {}
    for sentence in cleaned_text_list:
        for word in sentence.split():
            if word not in wc:
                wc[word] = 1
            else:
                wc[word] += 1
    
    return wc

def load_word2vec(path='P:/Pretrained Word2Vec Models/', fasttext=True):

    if fasttext:
        return pb.load(open(path + 'fasttext300d.pb', 'rb'))
    else:
        return pb.load(open(path + 'glove42B300d.pickle', 'rb'))

def create_embeddings_of_word2vec():

    embedding_index = load_word2vec()

    # for line in word2vec:
    #     values = line.split(' ')
    #     words = values[0]
    #     embedding_vector = np.asarray(values[1:], dtype='float32')
    #     embedding_index[words] = embedding_vector
    
    return embedding_index

def vocab_to_int(word_count, embedding_index):
    voc_to_int = {}
    value = 0
    for word, count in word_count.items():
        if word in embedding_index:
            voc_to_int[word] = value
            value += 1
        else:
            voc_to_int[word] = np.random.randint(2000, 2040)
    int_to_voc = {}
    for word, value in voc_to_int.items():
        int_to_voc[value] = word

    return voc_to_int, int_to_voc

def final_embedding_matrix(voc_to_int, embedding_index):
    embed_dim = 300
    nb_words = len(voc_to_int)

    word_embed_matrix = np.zeros((nb_words, embed_dim), dtype='float32')
    for word, i in voc_to_int.items():
        if word in embedding_index:
            word_embed_matrix[i] = embedding_index[word]
        else:
            new_embed = np.array(np.random.uniform(-1.0, 1.0, embed_dim))
            embedding_index[word] = new_embed
            word_embed_matrix[i] = new_embed
    
    return word_embed_matrix