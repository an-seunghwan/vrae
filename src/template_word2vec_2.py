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
body = pd.read_csv('./data/barycenter_body.csv')
body.head()
body.columns
#%%
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
template_dict = {x:i+1 for i,x in enumerate(sorted(list(set(body['sentence']))))}
template_dict['<PAD>'] = 0
len(template_dict)
template_numdict = {y:x for x,y in template_dict.items()}
#%%
userid_dict = {x:[] for x in sorted(list(set([str(x)+'_'+str(y) for x,y in zip(body['USER_ID'], body['CNSL_SN'])])))}
for i in tqdm(range(len(body))):
    id_ = '_'.join([str(x) for x in body[['USER_ID', 'CNSL_SN']].iloc[i].to_list()])
    userid_dict[id_] = userid_dict.get(id_) + [body['sentence'].iloc[i]]
userid_dict = {x:y for x,y in userid_dict.items() if len(y) > 1}
len(userid_dict)
#%%
corpus = [[template_dict.get(x) for x in sent] for sent in userid_dict.values()]
maxlen = max(len(x) for x in corpus)
window_size = maxlen-1
context_data = []
target_data = []
for j in tqdm(range(len(corpus))):
    sentence = corpus[j]
    for idx in range(len(sentence)):
        s = idx - window_size
        e = idx + window_size + 1
        context = []
        for i in range(s, e):
            if 0 <= i < len(sentence) and i != idx:
                context.append(sentence[i])
        target_data.append(sentence[idx])
        context_data.append(context)

context_input = preprocessing.sequence.pad_sequences(context_data, maxlen=maxlen, padding='post')
rows, cols = np.where(context_input != 0)
context_mask = np.zeros_like(context_input)
context_mask[rows, cols] = 1
context_mask = context_mask[:, np.newaxis, :]
context_mask.shape
target_input = np.array(target_data)[:, np.newaxis]
#%%
batch_size = 500
epochs = 100
embedding_size = 50

input_layer = layers.Input((maxlen))
mask_input = layers.Input((1, maxlen))
embedding_layer = layers.Embedding(len(template_dict), embedding_size)
h = tf.reduce_mean(tf.matmul(mask_input, embedding_layer(input_layer)), axis=1)
recon_layer = layers.Dense(len(template_dict), use_bias=False)
yhat = recon_layer(h)

model = K.models.Model([input_layer, mask_input], yhat)
model.summary()

model.compile('rmsprop',
              K.losses.SparseCategoricalCrossentropy(from_logits=True),
              ['accuracy'])

model.fit([context_input, context_mask], target_input,
          batch_size=batch_size,
          epochs=epochs)
#%%
# K.backend.clear_session()
#%%
embmat = embedding_layer.weights[0].numpy()[1:, :]
topk = 5
norm_ = np.linalg.norm(embmat, axis=1)
embedded_topk = []
for i in tqdm(range(embmat.shape[0])):
    temp = (embmat @ embmat[i]) / (norm_ * np.linalg.norm(embmat[i]))
    topkarg = np.argsort(temp)[-topk-1:-1][::-1]
    embedded_topk.append([template_numdict.get(i+1)] + [template_numdict.get(x) for x in topkarg])
#%% save
with open('./result/template_word2vec_200815.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['target', 'topk1', 'topk2', 'topk3', 'topk4', 'topk5'])
    for k in range(len(embedded_topk)): 
        wr.writerow([x.replace(u'\xa0', '') for x in embedded_topk[k]])
#%%





































