#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:29:42 2020

@author: anseunghwan
"""


#%%
'''
- latent 공간의 분리 - mixture distribution
- imputing missing words = 주어진 문장 형식에 빠진 부분 메꾸기
단, z의 정보를 반영하여 긍정, 부정 등의 특성을 나타내는 단어로 메꾸기
- beta(= sigma of continuous data) learning
- categorical reparametrization with gumbell softmax
- negative sampling?
- sentence interpolation
'''
#%%
import tensorflow as tf
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
# import math
# import json
import re
# import matplotlib.pyplot as plt
from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import os
# os.chdir('/home/jeon/Desktop/an/kakao_arena')
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%% data
'''한국은행 데이터'''
data = pd.read_csv('./data/total_sample_labeling_fin44.csv', encoding='euc-kr')
data.head()
data.columns
sent_idx = data['news/sentence'] == 0

'''감성라벨 데이터 추가 필요'''

sentence = data.loc[sent_idx]['content_new'].iloc[:100000].to_list()
#%% tokenize
p = re.compile('[가-힣]+')
useful_tag = ['Noun', 'Verb', 'Adjective', 'Adverb']
corpus = []
for i in tqdm(range(len(sentence))):
    if type(sentence[i] == str):
        corpus.append(['<sos>'] + [x[0] for x in okt.pos(sentence[i], stem=True) if p.match(x[0]) and len(x[0]) > 1 and x[1] in useful_tag] + ['<eos>'])
#%%
vocab = set()
for i in tqdm(range(len(corpus))):
    vocab.update(corpus[i])

vocab = {x:i+1 for i,x in enumerate(sorted(list(vocab)))}
vocab['UNK'] = 0
vocab_size = len(vocab)
print(len(vocab))

num_vocab = {i:x for x,i in vocab.items()}
#%%
input_text = [0]*len(corpus)
for i in tqdm(range(len(corpus))):
    input_text[i] = [vocab.get(x) for x in corpus[i]]
#%%
maxlen = max(len(x) for x in input_text)
input_text = preprocessing.sequence.pad_sequences(input_text,
                                                  maxlen=maxlen,
                                                  padding='post')
output_text = np.concatenate((input_text[:, 1:], np.zeros((len(input_text), 1))), axis=1)
#%% parameters
batch_size = 200
latent_dim = 100
embedding_size = 100
units = 100
#%% encoder
x = layers.Input((maxlen))
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size)
ex = embedding_layer(x)
encoder_gru = layers.GRU(units)
encoder_h = encoder_gru(ex)

mu_dense = layers.Dense(latent_dim)
log_var_dense = layers.Dense(latent_dim)
z_mean = mu_dense(encoder_h)
z_log_var = log_var_dense(encoder_h)
epsilon = tf.random.normal((batch_size, latent_dim))
z = z_mean + tf.math.exp(z_log_var / 2) * epsilon 
#%% decoder
# y = layers.Input((maxlen))
# ey = embedding_layer(y)
hiddens = layers.RepeatVector(maxlen)(z)
decoder_gru = layers.GRU(units, 
                         return_sequences=True)
'''for initial state, z could be reweighted using dense layer'''
decoder_h = decoder_gru(hiddens, initial_state=z)
logit_layer = layers.TimeDistributed(layers.Dense(vocab_size))
logit = logit_layer(decoder_h)
#%% model
text_vae = K.models.Model(x, [z_mean, z_log_var, z, logit])
text_vae.summary()
#%% loss
def loss_fun(y, y_pred, mean_pred, log_var_pred, beta):
    recon_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=False))
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + log_var_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(log_var_pred)), axis=1))
    return recon_loss, kl_loss, recon_loss + beta * kl_loss

# def loss_func(y, y_pred):
#     pos = tf.reduce_sum(tf.one_hot(y, depth=vocab_size+1), axis=1)[:, 1:]
#     neg = tf.cast(tf.logical_not(tf.cast(pos, dtype=tf.bool)), dtype=tf.float32)
#     loss_pos = -tf.reduce_mean(tf.multiply(tf.math.log(y_pred[:, 1:] + 1e-9), pos), keepdims=True)
#     loss_neg = -tf.reduce_mean(tf.multiply(tf.math.log(1 - y_pred[:, 1:] + 1e-9), neg), keepdims=True)
#     return tf.squeeze(loss_pos + loss_neg, axis=1)
#%%
'''
- kl annealing
- word dropout for encoder
'''
#%%
optimizer = tf.keras.optimizers.Adam(0.001)
#%% training 
epochs = 100
beta = 1
for epoch in range(epochs):

    idx = np.random.randint(0, len(input_text), batch_size) # sampling random batch -> stochasticity
    input_sequence = input_text[idx]    
    output_sequence = output_text[idx]
    
    '''word dropout with UNK'''
    
    with tf.GradientTape(persistent=True) as tape:
        
        # get output
        z_mean_pred, z_log_var_pred, z_pred, sequence_pred = text_vae(input_sequence)

        # ELBO 
        recon_loss, kl_loss, loss = loss_fun(output_sequence, sequence_pred, z_mean_pred, z_log_var_pred, beta)
        
    grad = tape.gradient(loss, text_vae.weights)
    optimizer.apply_gradients(zip(grad, text_vae.weights))

    if epoch % 10 == 0:
        print('({} epoch)'.format(epoch))
        print('Text VAE loss: {:.6}, Recon loss: {:.6}, KL: {:6}'.format(loss.numpy(), recon_loss.numpy(), kl_loss.numpy()))
#%%
# K.backend.clear_session()
#%%
inf_input = layers.Input((maxlen))
inf_output = 
inference_model


#%%





























































