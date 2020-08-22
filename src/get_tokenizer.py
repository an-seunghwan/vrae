# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:29:00 2019

@author: Hosik
"""

from konlpy.tag import Okt, Komoran, Hannanum, Kkma
from eunjeon import Mecab 
#tokenizer = get_tokenizer("mecab")
#tokenizer.morphs("아버지가방에들어가신다")

def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'komoran':
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    else:
        tokenizer = Mecab()
    return tokenizer

#tokenizer = get_tokenizer("komoran")
#tokenizer.morphs("아버지가방에들어가신다")
#tokenizer.pos("아버지가방에들어가신다")

# one-hot encoding
def one_hot_encoding(words, word2index):
    one_hot_vector = [0]*(len(word2index))
    for word in words:
         if word2index.get(word) != None:
             index = word2index[word]
             one_hot_vector[index] = 1
    return one_hot_vector

def make_x_one_hot_encoding(messages, word2index):
    x_one_hot = list() #np.zeros(len(messages), len(word2index))
    cnt = 0
    for message in messages:
        x_one_hot.append(one_hot_encoding(message, word2index))
    #return np.array(x_one_hot)
    return x_one_hot

# list encoding
def list_encoding(words, word2index):
    word2index_keys = word2index.keys()
    ##words = xc2_dfe['message'][2]
    #word2index_keys = word_to_index.keys()
    #words_s = words.split()
    mess_words = [token for token, tag in tokenizer.pos(words.replace(' ', '')) if tag=='NNG' or tag == 'NNP']
    word_list = [word for word in mess_words if word in word2index_keys]
    word_index = [word2index[word] for word in mess_words if word2index.get(word) != None]
    return word_list, word_index

def list_encoding2(words, word2index):
    one_hot_list = []
    for word in words:
         if word2index.get(word) != None:
             one_hot_list.append(word2index[word])
            
    return one_hot_list

def make_x_list_encoding(messages, word2index):
    x_one_hot = list() #np.zeros(len(messages), len(word2index))
    cnt = 0
    for message in messages:
        x_one_hot.append(one_hot_encoding(message, word2index))
    #return np.array(x_one_hot)
    return x_one_hot
    

































