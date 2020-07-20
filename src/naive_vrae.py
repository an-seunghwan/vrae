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
import math
import time
import re
import matplotlib.pyplot as plt
from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import os
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%% data 1
data = pd.read_csv('./data/1_구어체(1)_200226.csv', encoding='utf-8')
data.head()
data.columns
#%% tokenize
p = re.compile('[가-힣]+')
corpus = data['원문'].iloc[:100000].to_list()
for i in tqdm(range(len(corpus))):
    # corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(corpus[i], stem=False) if p.match(x[0]) and len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>']
    corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(corpus[i], stem=False) if len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>']
    # corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(corpus[i], stem=False) if p.match(x[0]) and len(x[0]) > 1] + ['<eos>']
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
maxlen = max(len(x) for x in input_text)
# maxlen = 50
input_text = preprocessing.sequence.pad_sequences(input_text,
                                                  maxlen=maxlen,
                                                  padding='post',
                                                  value=0)
output_text = np.concatenate((input_text[:, 1:], np.zeros((len(input_text), 1))), axis=1)
#%% parameters
batch_size = 500
embedding_size = 200
latent_dim = 100
units = 100
#%% encoder
x = layers.Input((maxlen))
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size)
ex = embedding_layer(x)
encoder_lstm = layers.LSTM(units)
encoder_h = encoder_lstm(ex)

mu_dense = layers.Dense(latent_dim)
log_var_dense = layers.Dense(latent_dim)
z_mean = mu_dense(encoder_h)
z_log_var = log_var_dense(encoder_h)
epsilon = tf.random.normal((latent_dim, ))
z = z_mean + tf.math.exp(z_log_var / 2) * epsilon 
#%% decoder
y = layers.Input((maxlen))
ey = embedding_layer(y)
decoder_lstm = layers.LSTM(units, 
                           return_sequences=True)

'''for initial state, z could be reweighted using dense layer'''
reweight_h_dense = layers.Dense(units)
reweight_c_dense = layers.Dense(units)
init_h = reweight_h_dense(z)
init_c = reweight_c_dense(z)

decoder_h = decoder_lstm(ey, initial_state=[init_h, init_c])
logit_layer = layers.TimeDistributed(layers.Dense(vocab_size))
logit = logit_layer(decoder_h)
#%% model
text_vae = K.models.Model([x, y], [z_mean, z_log_var, z, logit])
text_vae.summary()
#%% loss
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)
def loss_fun(y, y_pred, mean_pred, log_var_pred, beta):
    '''do not consider padding'''
    # reconstruction loss
    non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(scce(y, y_pred), 
                                                                    tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
                                                        non_pad_count), axis=1))
    
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + log_var_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(log_var_pred)), axis=1))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%% kl annealing
'''
- kl annealing
'''
def kl_anneal(step, s, k=0.001):
    return 1 / (1 + math.exp(-k*(step - s)))
#%% optimizer
optimizer = tf.keras.optimizers.Adam(0.005)
#%% training 
epochs = 2000
dropout_rate = 0.5
for epoch in range(1, epochs):
    beta = kl_anneal(epoch, epochs)
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
latent_h = encoder_lstm(latent_emb)
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
latent_h = reweight_h_dense(inf_hidden)
latent_c = reweight_c_dense(inf_hidden)
inf_output = logit_layer(decoder_lstm(inf_emb, initial_state=[latent_h, latent_c]))
inference_model = K.models.Model([inf_input, inf_hidden], inf_output)
inference_model.summary()
#%% interpolation & inference
j1 = 200
j2 = 201
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





























































