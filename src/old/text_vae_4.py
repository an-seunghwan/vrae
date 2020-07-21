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
import math
# import json
import time
import re
import matplotlib.pyplot as plt
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
#%% data 1
'''한국은행 데이터'''
data = pd.read_csv('./data/total_sample_labeling_fin44.csv', encoding='euc-kr')
data.head()
data.columns
sentence_idx = data['news/sentence'] == 0
#%% data 2
'''감성라벨 데이터'''
# 소비자와 기업을 모두 사용
sentiment_idx1 = data['소비자'] != 0 # 경제심리가 있는 문장만을 추출
sentiment_idx2 = data['기업'] != 0 # 경제심리가 있는 문장만을 추출
sentence1 = data.loc[sentence_idx & sentiment_idx1]['content_new'].to_list()
sentence2 = data.loc[sentence_idx & sentiment_idx2]['content_new'].to_list()
sentence = sentence1 + sentence2
print(len(sentence))
#%%
'''.(마침표)를 단위로 다시 문장을 분리한다(기사가 껴있는 경우가 있어 이를 방지)'''
corpus = []
for i in tqdm(range(len(sentence))):
    # corpus.extend([x + '.' for x in sentence[i].split('. ')])
    corpus.extend([x.strip() for x in sentence[i].split('. ')])
#%% tokenize
p = re.compile('[가-힣]+')
# useful_tag = ['Noun', 'Verb', 'Adjective', 'Adverb']
for i in tqdm(range(len(corpus))):
    if type(corpus[i] == str):
        # corpus.append(['<sos>'] + [x[0] for x in okt.pos(sentence[i], stem=True) if p.match(x[0]) and len(x[0]) > 1 and x[1] in useful_tag] + ['<eos>'])
        corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(corpus[i], stem=True) if p.match(x[0]) and len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>']
#%%
vocab = set()
for i in tqdm(range(len(corpus))):
    vocab.update(corpus[i])

vocab = {x:i+2 for i,x in enumerate(sorted(list(vocab)))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1
vocab_size = len(vocab)
print(len(vocab))

num_vocab = {i:x for x,i in vocab.items()}
#%%
input_text = [0]*len(corpus)
for i in tqdm(range(len(corpus))):
    input_text[i] = [vocab.get(x) for x in corpus[i]]
#%%
# maxlen 결정
plt.hist([len(x) for x in corpus])
# maxlen = max(len(x) for x in input_text)
maxlen = 50
input_text = preprocessing.sequence.pad_sequences(input_text,
                                                  maxlen=maxlen,
                                                  padding='post',
                                                  value=0)
output_text = np.concatenate((input_text[:, 1:], np.zeros((len(input_text), 1))), axis=1)
#%% parameters
batch_size = 200
embedding_size = 100
latent_dim = 100
units = 100
#%% encoder
x = layers.Input((maxlen))
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size)
ex = embedding_layer(x)
encoder_lstm = layers.LSTM(units)
encoder_h = encoder_lstm(ex)
'''Gaussian mixture'''
mu_dense1 = layers.Dense(latent_dim)
log_var_dense1 = layers.Dense(latent_dim)
mu_dense2 = layers.Dense(latent_dim)
log_var_dense2 = layers.Dense(latent_dim)

z_mean1 = mu_dense1(encoder_h)
z_log_var1 = log_var_dense1(encoder_h)
z_mean2 = mu_dense2(encoder_h)
z_log_var2 = log_var_dense2(encoder_h)

epsilon1 = tf.random.normal((batch_size, latent_dim))
epsilon2 = tf.random.normal((batch_size, latent_dim))
z1 = z_mean1 + tf.math.exp(z_log_var1 / 2) * epsilon1 
z2 = z_mean2 + tf.math.exp(z_log_var2 / 2) * epsilon2 
#%% decoder
y = layers.Input((maxlen))
ey = embedding_layer(y)
# hiddens = layers.RepeatVector(maxlen)(z)
decoder_gru = layers.GRU(units, 
                         return_sequences=True)
'''for initial state, z could be reweighted using dense layer'''
decoder_h = decoder_gru(ey, initial_state=z)
logit_layer = layers.TimeDistributed(layers.Dense(vocab_size))
logit = logit_layer(decoder_h)
#%% model
text_vae = K.models.Model([x, y], [z_mean, z_log_var, z, logit])
text_vae.summary()
#%% loss
def loss_fun(y, y_pred, mean_pred, log_var_pred, beta):
    recon_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, y_pred, from_logits=True))
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
'''
def kl_anneal(step, s, k=0.01):
    return 1 / (1 + math.exp(-k*(step - s)))
#%%
optimizer = tf.keras.optimizers.Adam(0.005)
#%% training 
epochs = 1200
# beta = 0.1
dropout_rate = 0.5
for epoch in range(1, epochs):
    beta = kl_anneal(epoch, epochs/2)
    if epoch % 10 == 1:
        t1 = time.time()

    idx = np.random.randint(0, len(input_text), batch_size) # sampling random batch -> stochasticity
    input_sequence = input_text[idx]    
    input_sequence_dropout = input_text[idx]    
    output_sequence = output_text[idx]
    
    '''word dropout with UNK'''
    non_pad = np.sum(input_sequence != vocab.get('<PAD>'), axis=1)
    dropout_ = [np.random.binomial(1, dropout_rate, x-1) for x in non_pad]
    dropout_index = [d  * np.arange(x-1) for d, x in zip(dropout_, non_pad)]
    for i in range(batch_size):
        input_sequence_dropout[i][[d for d in dropout_index[i] if d != 0]] = vocab.get('<UNK>')
    
    with tf.GradientTape(persistent=True) as tape:
        
        # get output
        z_mean_pred, z_log_var_pred, z_pred, sequence_pred = text_vae([input_sequence, input_sequence_dropout])

        # ELBO 
        recon_loss, kl_loss, loss = loss_fun(output_sequence, sequence_pred, z_mean_pred, z_log_var_pred, beta)
        
    grad = tape.gradient(loss, text_vae.weights)
    optimizer.apply_gradients(zip(grad, text_vae.weights))

    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3})'.format(epoch, t2-t1))
        print('Text VAE loss: {:.6}, Reconstruction: {:.6}, KL: {:.6}'.format(loss.numpy(), recon_loss.numpy(), kl_loss.numpy()))
#%%
# K.backend.clear_session()
#%% latent generation
latent_input = layers.Input((maxlen))
latent_emb = embedding_layer(latent_input)
latent_h = encoder_gru(latent_emb)
latent_mean = mu_dense(latent_h)
latent_log_var = log_var_dense(latent_h)
epsilon = tf.random.normal((latent_dim, ))
latent_z = latent_mean + tf.math.exp(latent_log_var / 2) * epsilon 
latent_model = K.models.Model(latent_input, latent_z)
latent_model.summary()
#%% inference model
inf_input = layers.Input((maxlen))
inf_emb = embedding_layer(inf_input)
inf_hidden = layers.Input((latent_dim))
inf_output = logit_layer(decoder_gru(inf_emb, initial_state=inf_hidden))
inference_model = K.models.Model([inf_input, inf_hidden], inf_output)
inference_model.summary()
#%% interpolation & inference
j1 = 10
j2 = 100
print('===input===')
print(' '.join([num_vocab.get(x) for x in input_text[j1, :] if x != 0]))
print(' '.join([num_vocab.get(x) for x in input_text[j2, :] if x != 0]))
z1 = latent_model(input_text[[j1], :])
z2 = latent_model(input_text[[j2], :])

# interpolation
z_inter = z1
for v in np.linspace(0, 1, 7):
    z_inter = np.vstack((z_inter, v * z1 + (1 - v) * z2))
z_inter = np.vstack((z_inter, z2))

val_seq = np.zeros((len(z_inter), maxlen))
val_seq[:, 0] = vocab.get('<sos>')
result = ['']*len(z_inter)

for k in range(len(result)):
    for t in range(1, maxlen):
        pred = inference_model([val_seq[[k], :], z_inter[[k], :]])
        pred_id = tf.argmax(pred[0][t-1]).numpy()
        result[k] += num_vocab.get(pred_id) + ' '
        
        if num_vocab.get(pred_id) == '<eos>':
            break
    
        val_seq[:, t] = pred_id
print('===output===')
pprint(result)
#%%





























































