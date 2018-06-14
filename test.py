import pandas as pd   
from keras.preprocessing.text import Tokenizer
import utils

df = pd.read_csv('./Data/yelp_dataset.txt', sep='\t', header=None)

all_text = df[0].values.tolist()
print(type(all_text))

new_text = []
for l in all_text:
    text = utils.clean_text(l)
    new_text.append(text)

tokenizer = Tokenizer(num_words=30000)
tokenizer.fit_on_texts(list(new_text))

print(len(tokenizer.word_index))

print(new_text[:5])