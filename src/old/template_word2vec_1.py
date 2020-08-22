#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:10:43 2020

@author: anseunghwan
"""

#%%
import tensorflow as tf
# import tensorflow_probability as tfp
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import pandas as pd
import numpy as np
# import random
import math
# import json
import csv    
import time
import re
# import matplotlib.pyplot as plt
# from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import pickle
# import sklearn
import os
# os.chdir('/home/jeon/Desktop/an/kakao_arena')
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%%
'''barycenter(template) data'''
body = pd.read_csv('./data/body_template.csv')
body.head()
body.columns
semantic = body[['sentence', '시맨틱']]
semantic_dict = {x:y for x,y in zip(semantic['sentence'], semantic['시맨틱'])}
# 중복된 행 제거
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
#%%
corpus = list(set(body['sentence'].to_list()))
tokenized = [0]*len(corpus)
p = re.compile('[가-힣]+')
for i in tqdm(range(len(tokenized))):
    tokenized[i] = [x[0] for x in okt.pos(corpus[i], stem=False) if p.match(x[0]) and len(x[0]) > 1 and x[1] != 'Josa']
#%%
vocab = set()
for i in tqdm(range(len(tokenized))):
    vocab.update(tokenized[i])
vocab = {x:i+1 for i,x in enumerate(sorted(list(vocab)))}
vocab['<PAD>'] = 0
vocab_size = len(vocab)
num2word = {y:x for x,y in vocab.items()}
#%%
window_size = 4
context_data = []
target_data = []
for j in tqdm(range(len(tokenized))):
    sentence = tokenized[j]
    for idx in range(len(sentence)):
        s = idx - window_size
        e = idx + window_size + 1
        context = []
        for i in range(s, e):
            if 0 <= i < len(sentence) and i != idx:
                context.append(vocab.get(sentence[i]))
        target_data.append(vocab.get(sentence[idx]))
        context_data.append(context)

context_input = preprocessing.sequence.pad_sequences(context_data, maxlen=window_size*2, padding='post')
target_input = np.array(target_data)[:, np.newaxis]
#%%
batch_size = 500
epochs = 100
embedding_size = 50

input_layer = layers.Input((window_size*2))
embedding_layer = layers.Embedding(vocab_size, embedding_size)
h = layers.GlobalAveragePooling1D()(embedding_layer(input_layer))
recon_layer = layers.Dense(vocab_size, use_bias=False)
yhat = recon_layer(h)

model = K.models.Model(input_layer, yhat)
model.summary()

model.compile('rmsprop',
              K.losses.SparseCategoricalCrossentropy(from_logits=True),
              ['accuracy'])

model.fit(context_input, target_input,
          batch_size=batch_size,
          epochs=epochs)
#%%
# K.backend.clear_session()
#%%
embmat = embedding_layer.weights[0].numpy()
#%%
tokenized_dict = {i:x for i,x in enumerate(tokenized)}
embedded_dict = {}
for i in tqdm(range(len(tokenized))):
    embedded_dict[i] = np.mean(embmat[[vocab.get(x) for x in tokenized[i]], :], axis=0, keepdims=True)
embedded_total = np.array(list(embedded_dict.values())).reshape(len(tokenized), embedding_size)
#%%
from sklearn.cluster import KMeans
#%%
kmeans = KMeans(n_clusters=8)
#model = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4)
kmeans.fit(embedded_total)

kmeans_dict = {}
for i in range(len(kmeans.labels_)):
    kmeans_dict[kmeans.labels_[i]] = kmeans_dict.get(kmeans.labels_[i], []) + [i]
#%%
template_group_dict = {}
for label in range(8):
    template_group_dict[label] = sorted([' '.join(tokenized_dict.get(x)) for x in kmeans_dict.get(label)])
#%%
maxlen_ = max(len(x) for x in template_group_dict.values())
for label in range(8):
    template_group_dict[label] = template_group_dict.get(label) + [' ']*(maxlen_ - len(template_group_dict.get(label)))
#%%
temp = pd.DataFrame.from_dict(template_group_dict)
temp.to_csv('./result/template_clustering.csv', encoding='euc-kr')
#%%
# topk = 5
# norm_ = np.linalg.norm(embedded_total, axis=1)
# embedded_topk_dict = {}
# for i in tqdm(range(len(tokenized))):
#     temp = np.squeeze(embedded_total @ embedded_dict[i].T)
#     temp = temp / (norm_ * np.linalg.norm(embedded_dict[i]))   
#     embedded_topk_dict[i] = list(np.argsort(temp)[-topk-1:-1][::-1])
#%%


#%%





































