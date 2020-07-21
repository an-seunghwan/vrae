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
data = data.loc[sentence_idx].reset_index()
#%% data 2
'''감성라벨 데이터'''
# 소비자와 기업을 모두 사용
sentiment_idx1_pos = np.array(data['소비자'] == 1) | np.array(data['소비자'] == 4)
sentiment_idx1_neg = np.array(data['소비자'] == 2) | np.array(data['소비자'] == 5)

sentiment_idx2_pos = np.array(data['기업'] == 1) | np.array(data['기업'] == 4)
sentiment_idx2_neg = np.array(data['기업'] == 2) | np.array(data['기업'] == 5)

sentence1_pos = data.loc[sentiment_idx1_pos]['content_new'].to_list()
sentence1_neg = data.loc[sentiment_idx1_neg]['content_new'].to_list()

sentence2_pos = data.loc[sentiment_idx2_pos]['content_new'].to_list()
sentence2_neg = data.loc[sentiment_idx2_neg]['content_new'].to_list()

sentence = sentence1_pos + sentence2_pos + sentence1_neg + sentence2_neg
print(len(sentence))

# label
label_ = np.zeros((len(sentence), 2))
label_[:len(sentence1_pos + sentence2_pos), 0] = 1
label_[len(sentence1_pos + sentence2_pos):, 1] = 1
#%%
'''.(마침표)를 단위로 다시 문장을 분리한다(기사가 껴있는 경우가 있어 이를 방지)'''
corpus_ = []
label_data = []
for i in tqdm(range(len(sentence))):
    # corpus.extend([x + '.' for x in sentence[i].split('. ')])
    temp = [x.strip() for x in sentence[i].split('. ')]
    corpus_.extend(temp)
    label_data.extend([label_[i] for _ in range(len(temp))])
#%%
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub('', sent)
    else:
        result = ''
    return result
#%% tokenize
p = re.compile('[가-힣]+')
corpus = []
label = []
# useful_tag = ['Noun', 'Verb', 'Adjective', 'Adverb']
for i in tqdm(range(len(corpus_))):
    if type(corpus_[i] == str):
        # corpus.append(['<sos>'] + [x[0] for x in okt.pos(sentence[i], stem=True) if p.match(x[0]) and len(x[0]) > 1 and x[1] in useful_tag] + ['<eos>'])
        # corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(temp, stem=False) if p.match(x[0]) and len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>']
        temp = clean_korean(corpus_[i])
        corpus.append(['<sos>'] + [x[0] for x in okt.pos(temp, stem=False) if len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>'])
        label.append(label_data[i])
label = np.array(label)
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
embedding_size = 150
latent_dim = 40
units = 100
#%% prior
M = 2 # the number of components

prior_mu = np.ones((M, latent_dim))
prior_mu[0, :] *= 2
prior_mu[1, :] *= -2

'''we set sigma for 1 globally'''
#%% encoder
x = layers.Input((maxlen))
# embedding_layer = layers.Embedding(input_dim=vocab_size,
#                                     output_dim=embedding_size)
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                    output_dim=embedding_size,
                                    mask_zero=True)
ex = embedding_layer(x)
encoder_lstm = layers.LSTM(units)
encoder_h = encoder_lstm(ex)

# for 문으로 list로 생성(나중에 M이 커지면?)
mix_prob_dense = layers.Dense(M, activation='softmax')
mean_dense1 = layers.Dense(latent_dim)
log_var_dense1 = layers.Dense(latent_dim)
mean_dense2 = layers.Dense(latent_dim)
log_var_dense2 = layers.Dense(latent_dim)

mix_prob = mix_prob_dense(encoder_h)
z_mean1 = mean_dense1(encoder_h)
z_log_var1 = log_var_dense1(encoder_h)
z_mean2 = mean_dense2(encoder_h)
z_log_var2 = log_var_dense2(encoder_h)

prob_sampling = tf.random.categorical(mix_prob, 1)
chosen_idx = tf.concat((prob_sampling, tf.cast(tf.cast(tf.logical_not(tf.cast(prob_sampling, tf.bool)), tf.bool), tf.int64)), axis=1)

epsilon1 = tf.random.normal((latent_dim, ))
z1 = z_mean1 + tf.math.exp(z_log_var1 / 2) * epsilon1
epsilon2 = tf.random.normal((latent_dim, ))
z2 = z_mean2 + tf.math.exp(z_log_var2 / 2) * epsilon2

z12 = tf.concat((z1[:, tf.newaxis, :], z2[:, tf.newaxis, :]), axis=1)
z = tf.reduce_sum(tf.multiply(tf.cast(tf.tile(chosen_idx[..., tf.newaxis], (1, 1, latent_dim)), tf.float32), z12), axis=1)
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
logit_layer = layers.TimeDistributed(layers.Dense(vocab_size)) # no softmax normalizaing -> logit tensor (from_logits=True)
logit = logit_layer(decoder_h)
#%% model
mixprob_vae = K.models.Model([x, y], mix_prob)
mixprob_vae.summary()
text_vae = K.models.Model([x, y], [z_mean1, z_log_var1, z_mean2, z_log_var2, logit])
text_vae.summary()  
#%% decoder by case
# case1 = True
# # case1 = False

# if case1:
#     '''decoder case 1: latent variable z in only given as hidden vector of LSTM'''
#     y = layers.Input((maxlen))
#     ey = embedding_layer(y)
#     decoder_lstm = layers.LSTM(units, 
#                                return_sequences=True)
#     '''for initial state, z could be reweighted using dense layer'''
#     decoder_h = decoder_lstm(ey, initial_state=[z, z])
#     logit_layer = layers.TimeDistributed(layers.Dense(vocab_size))
#     logit = logit_layer(decoder_h)
    
#     text_vae = K.models.Model([x, y], [z_mean, z_log_var, z, logit])
#     text_vae.summary()  
# else:
#     '''decoder case 2: latent variable z in given as input of decoder
#     in this case, word dropout is not needed'''
#     hiddens = layers.RepeatVector(maxlen)(z)
#     decoder_lstm = layers.LSTM(units, 
#                                return_sequences=True)
#     '''for initial state, z could be reweighted using dense layer'''
#     decoder_h = decoder_lstm(hiddens, initial_state=[z, z])
#     logit_layer = layers.TimeDistributed(layers.Dense(vocab_size))
#     logit = logit_layer(decoder_h)
    
#     text_vae = K.models.Model(x, [z_mean, z_log_var, z, logit])
#     text_vae.summary()
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
    # non_pad_ = np.sum(y != vocab.get('<PAD>'), axis=1)
    # recon_loss = tf.zeros(())
    # for i in range(len(non_pad_)):
    #     n = non_pad_[i]
    #     recon_loss += scce(y[[i], :n], y_pred[i, :n, :]) / n
    # recon_loss /= len(non_pad_)
    
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + log_var_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(log_var_pred)), axis=1))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%% loss 
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)
def loss_mixture_fun(y, y_pred, mean_pred1, log_var_pred1, mean_pred2, log_var_pred2, pi, beta):
    '''do not consider padding'''
    # reconstruction loss
    non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(scce(y, y_pred), 
                                                                    tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
                                                        non_pad_count), axis=1))
    
    # kl-divergence loss
    term1 = tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi * M), axis=1))
    kl1 = tf.reduce_sum(-0.5 * (1 + log_var_pred1 - tf.math.pow(mean_pred1 - prior_mu[0, :], 2) - tf.math.exp(log_var_pred1)), axis=1, keepdims=True)
    kl2 = tf.reduce_sum(-0.5 * (1 + log_var_pred2 - tf.math.pow(mean_pred2 - prior_mu[1, :], 2) - tf.math.exp(log_var_pred2)), axis=1, keepdims=True)
    kl_loss = term1 + tf.reduce_mean(tf.reduce_sum(tf.multiply(pi, tf.concat((kl1, kl2), axis=1)), axis=1))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.001):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%%
optimizer = tf.keras.optimizers.Adam(0.005)
optimizer1 = tf.keras.optimizers.Adam(0.005)
#%% training 
epochs = 3000
# beta = 0.1
dropout_rate = 0.5
for epoch in range(700, epochs):
    beta = kl_anneal(epoch, int(epochs/2))
    if epoch % 10 == 1:
        t1 = time.time()

    idx = np.random.randint(0, len(input_text), batch_size) # sampling random batch -> stochasticity
    input_sequence = input_text[idx][:, ::-1]
    input_sequence_dropout = input_text[idx]    
    output_sequence = output_text[idx]
    
    '''word dropout with UNK
    -> hold PAD and UNK word embedding vector zero vector(non-trainable)'''
    non_pad = np.sum(input_sequence != vocab.get('<PAD>'), axis=1)
    dropout_ = [np.random.binomial(1, dropout_rate, x-2) for x in non_pad]
    dropout_index = [d  * np.arange(1, x-1) for d, x in zip(dropout_, non_pad)]
    for i in range(batch_size):
        input_sequence_dropout[i][[d for d in dropout_index[i] if d != 0]] = vocab.get('<UNK>')
    
    with tf.GradientTape(persistent=True) as tape:
        
        # get output
        z_mean_pred1, z_log_var_pred1, z_mean_pred2, z_log_var_pred2, sequence_pred = text_vae([input_sequence, input_sequence_dropout])
        pi_hat = mixprob_vae([input_sequence, input_sequence_dropout])
        
        # ELBO 
        recon_loss, kl_loss, loss = loss_mixture_fun(output_sequence, sequence_pred, z_mean_pred1, z_log_var_pred1, z_mean_pred2, z_log_var_pred2, pi_hat, beta)
        
        # mixture probability loss
        mix_loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.multiply(label[idx, :], pi_hat), axis=1)))
        
    grad = tape.gradient(loss, text_vae.weights)
    optimizer.apply_gradients(zip(grad, text_vae.weights))
    grad1 = tape.gradient(mix_loss, mixprob_vae.weights)
    optimizer1.apply_gradients(zip(grad1, mixprob_vae.weights))

    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3})'.format(epoch, t2-t1))
        print('Text VAE loss: {:.6}, Reconstruction: {:.6}, KL: {:.6}, MIX: {:.6}'.format(loss.numpy(), recon_loss.numpy(), kl_loss.numpy(), mix_loss.numpy()))
#%%
# K.backend.clear_session()
#%% latent generation
latent_input = layers.Input((maxlen))
latent_emb = embedding_layer(latent_input)
latent_h = encoder_lstm(latent_emb)

latent_mix_prob = mix_prob_dense(latent_h)
latent_mean1 = mean_dense1(latent_h)
latent_log_var1 = log_var_dense1(latent_h)
latent_mean2 = mean_dense2(latent_h)
latent_log_var2 = log_var_dense2(latent_h)

latent_prob_sampling = tf.random.categorical(latent_mix_prob, 1)
latent_chosen_idx = tf.concat((latent_prob_sampling, tf.cast(tf.cast(tf.logical_not(tf.cast(latent_prob_sampling, tf.bool)), tf.bool), tf.int64)), axis=1)

epsilon1 = tf.random.normal((latent_dim, ))
latent_z1 = latent_mean1 + tf.math.exp(latent_log_var1 / 2) * epsilon1
epsilon2 = tf.random.normal((latent_dim, ))
latent_z2 = latent_mean2 + tf.math.exp(latent_log_var2 / 2) * epsilon2

latent_z12 = tf.concat((latent_z1[:, tf.newaxis, :], latent_z2[:, tf.newaxis, :]), axis=1)
latent_z = tf.reduce_sum(tf.multiply(tf.cast(tf.tile(latent_chosen_idx[..., tf.newaxis], (1, 1, latent_dim)), tf.float32), latent_z12), axis=1)

latent_model = K.models.Model(latent_input, latent_z)
latent_model.summary()
#%% inference model
inf_input = layers.Input((maxlen))
inf_hidden = layers.Input((latent_dim))
inf_emb = embedding_layer(inf_input) 
latent_init_h = reweight_h_dense(inf_hidden)
latent_init_c = reweight_c_dense(inf_hidden)
inf_output = logit_layer(decoder_lstm(inf_emb, initial_state=[latent_init_h, latent_init_c]))
inference_model = K.models.Model([inf_input, inf_hidden], inf_output)
inference_model.summary()
#%% interpolation & inference
j1 = 2
j2 = 3
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





























































