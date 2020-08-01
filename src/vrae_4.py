#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:29:42 2020

@author: anseunghwan
"""


#%%
'''
=== attention vrae(bidirectional) ===
- using whole article(not sentence)
- tanh activation for variance learning
- min word frequence >= 5
- top-k sampling for inference
- with teacher forcing 
- do not consider <PAD> for loss
'''
#%%
'''
=== what to do? ===
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
# from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import os
# os.chdir('/home/jeon/Desktop/an/kakao_arena')
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
# %% data 1
'''한국은행 데이터 -> 기사 전체를 사용'''
data = pd.read_csv('./data/total_sample_labeling_fin44.csv', encoding='euc-kr')
data.head()
data.columns
# 기사 데이터
# sentence_idx = data['news/sentence'] == 0
article_idx = data['news/sentence'] == 1
data = data.loc[article_idx].reset_index()
#%% data 2
'''감성라벨 데이터(기사의 경우)'''
# # 소비자와 기업을 모두 사용
sentiment_idx1_pos = np.array(data['소비자'] == 1) | np.array(data['소비자'] == 4)
sentiment_idx1_neg = np.array(data['소비자'] == 2) | np.array(data['소비자'] == 5)

sentiment_idx2_pos = np.array(data['기업'] == 1) | np.array(data['기업'] == 4)
sentiment_idx2_neg = np.array(data['기업'] == 2) | np.array(data['기업'] == 5)

article1_pos = data.loc[sentiment_idx1_pos]['content_new'].to_list()
article1_neg = data.loc[sentiment_idx1_neg]['content_new'].to_list()

article2_pos = data.loc[sentiment_idx2_pos]['content_new'].to_list()
article2_neg = data.loc[sentiment_idx2_neg]['content_new'].to_list()

article = article1_pos + article2_pos + article1_neg + article2_neg
print(len(article))

# label
# label_ = np.zeros((len(article), 2))
# label_[:len(article1_pos + article2_pos), 0] = 1
# label_[len(article1_pos + article2_pos):, 1] = 1
#%%
'''.(마침표)를 단위로 다시 문장(혹은 기사)을 분리한다(기사가 껴있는 경우가 있어 이를 방지)'''
corpus_ = []
# label_data = []
# for i in tqdm(range(len(sentence))):
for i in tqdm(range(len(article))):
    # temp = [x.strip() for x in sentence[i].split('. ')]
    temp = [x.strip() for x in article[i].split('. ')]
    corpus_.extend(temp)
    # label_data.extend([label_[i] for _ in range(len(temp))])
#%%
def clean_korean(sent):
    if type(sent) == str:
        h = re.compile('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]+')
        result = h.sub('', sent)
    else:
        result = ''
    return result

def clean_special_token_numbers(sent):
    if type(sent) == str:
        h1 = re.compile('[^\w\s]')
        h2 = re.compile('[0-9]')
        result = h1.sub('', sent)
        result = h2.sub('', result)
    else:
        result = ''
    return result
#%% tokenize
# p = re.compile('[가-힣]+')
corpus = []
# label = []
# useful_tag = ['Noun', 'Verb', 'Adjective', 'Adverb']
for i in tqdm(range(len(corpus_))):
    temp = clean_special_token_numbers(corpus_[i])
        
    if len(temp):
        # corpus.append(['<sos>'] + [x[0] for x in okt.pos(sentence[i], stem=True) if p.match(x[0]) and len(x[0]) > 1 and x[1] in useful_tag] + ['<eos>'])
        # corpus[i] = ['<sos>'] + [x[0] for x in okt.pos(temp, stem=False) if p.match(x[0]) and len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>']
        corpus.append(['<sos>'] + [x[0]+x[1] for x in okt.pos(temp, stem=False) if len(x[0]) > 1 and x[1] != 'Josa'] + ['<eos>'])
    
        # label.append(label_data[i])
# label = np.array(label)
#%% save corpus
import csv    
with open('./data/corpus.csv', 'w', encoding='utf-8', newline='') as f:
    wr = csv.writer(f)
    for i in tqdm(range(len(corpus))):
        wr.writerow([' '.join(corpus[i])])
#%% load corpus
corpus_ = []
with open('./data/corpus.csv', 'r', encoding='utf-8') as f:
    corpus_ = f.readlines()
corpus = [x.split() for x in corpus_]
#%% build vocab
corpus_ = []
for i in range(len(corpus)):
    corpus_.extend(corpus[i])

from collections import Counter
freq = Counter(corpus_)
plt.hist([x for x in list(freq.values()) if x <= 50])

vocab = []
for x, y in freq.items():
    if y >= 5:
        vocab.append(x)

vocab = {x:i+2 for i,x in enumerate(sorted(vocab))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

vocab_size = len(vocab)
print(len(vocab))

num_vocab = {i:x for x,i in vocab.items()}
#%%
input_text = [0]*len(corpus)
for i in tqdm(range(len(corpus))):
    # input_text[i] = [vocab.get(x, 1) for x in corpus[i]] # UNK 포함
    input_text[i] = [vocab.get(x) for x in corpus[i] if x in vocab]
#%%
# maxlen 결정
plt.hist([len(x) for x in corpus])
# maxlen = max(len(x) for x in input_text)
maxlen = 100
input_text = preprocessing.sequence.pad_sequences(input_text,
                                                  maxlen=maxlen,
                                                  padding='post',
                                                  value=0)
# output_text = np.concatenate((input_text[:, 1:], np.zeros((len(input_text), 1))), axis=1)
output_text = input_text[:, 1:]
#%% parameters
batch_size = 200
embedding_size = 150
latent_dim = 64
units = 64
#%% prior
# M = 2 # the number of components

# prior_mu = np.ones((M, latent_dim))
# prior_mu[0, :] *= 3
# prior_mu[1, :] *= -3

# '''we set sigma for 1 globally'''
#%% encoder
x = layers.Input((maxlen))
# label_input = layers.Input((M))
# embedding_layer = layers.Embedding(input_dim=vocab_size,
#                                     output_dim=embedding_size)
embedding_layer = layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size,
                                   mask_zero=True)

ex = embedding_layer(x)
# encoder_lstm = layers.LSTM(units)
encoder_lstm = layers.Bidirectional(layers.LSTM(units))
encoder_h = encoder_lstm(ex)

# for 문으로 list로 생성(나중에 M이 커지면?)
# mix_prob_dense = layers.Dense(M, activation='softmax')
mean_dense1 = layers.Dense(latent_dim)
# log_var_dense1 = layers.Dense(latent_dim, activation='softplus')
log_var_dense1 = layers.Dense(latent_dim, activation='tanh')
# mean_dense2 = layers.Dense(latent_dim)
# log_var_dense2 = layers.Dense(latent_dim)

# mix_prob = mix_prob_dense(encoder_h)
z_mean1 = mean_dense1(encoder_h)
z_log_var1 = log_var_dense1(encoder_h)
# z_mean2 = mean_dense2(encoder_h)
# z_log_var2 = log_var_dense2(encoder_h)

'''tf probability distribution으로 대체 가능?'''
# prob_sampling = tf.random.categorical(mix_prob, 1)
# chosen_idx = tf.concat((prob_sampling, tf.cast(tf.cast(tf.logical_not(tf.cast(prob_sampling, tf.bool)), tf.bool), tf.int64)), axis=1)

# epsilon1 = tf.random.normal((latent_dim, ))
# z1 = z_mean1 + tf.math.exp(z_log_var1 / 2) * epsilon1
# epsilon2 = tf.random.normal((latent_dim, ))
# z2 = z_mean2 + tf.math.exp(z_log_var2 / 2) * epsilon2

# z12 = tf.concat((z1[:, tf.newaxis, :], z2[:, tf.newaxis, :]), axis=1)
# z = tf.reduce_sum(tf.multiply(tf.cast(tf.tile(chosen_idx[..., tf.newaxis], (1, 1, latent_dim)), tf.float32), z12), axis=1)

'''label을 input으로 줄 때'''
epsilon1 = tf.random.normal((latent_dim, ))
z1 = z_mean1 + tf.math.exp(z_log_var1 / 2) * epsilon1
# epsilon2 = tf.random.normal((latent_dim, ))
# z2 = z_mean2 + tf.math.exp(z_log_var2 / 2) * epsilon2

# z12 = tf.concat((z1[:, tf.newaxis, :], z2[:, tf.newaxis, :]), axis=1)
# z = tf.reduce_sum(tf.multiply(z12, tf.tile(label_input[..., tf.newaxis], (1, 1, latent_dim))), axis=1)
#%% decoder
# y = layers.Input((maxlen-1))
# ey = embedding_layer(y)
decoder_lstm = layers.LSTM(units, 
                           return_sequences=True)
'''concatenate latent to decoder input'''
hiddens = layers.RepeatVector(maxlen-1)(z1)
# decoder_output = decoder_lstm(tf.concat((ey, hiddens), axis=-1)) 
decoder_output = decoder_lstm(hiddens, initial_state=[z1, z1]) 

'''for initial state, z could be reweighted using dense layer'''
'''reweight is too strong...?'''
# reweight_h_dense = layers.Dense(units)
# reweight_c_dense = layers.Dense(units)
# init_h = reweight_h_dense(z)
# init_c = reweight_c_dense(z)

logit_layer = layers.TimeDistributed(layers.Dense(vocab_size)) # no softmax normalizaing -> logit tensor (from_logits=True)
# logit_layer = layers.Dense(vocab_size)
logit = logit_layer(decoder_output)
#%% model
# mixprob_vae = K.models.Model(x, mix_prob)
# mixprob_vae.summary()
# text_vae = K.models.Model([x, label_input, y], [z_mean1, z_log_var1, z_mean2, z_log_var2, logit])
# text_vae = K.models.Model([x, y], [z_mean1, z_log_var1, logit])
text_vae = K.models.Model(x, [z_mean1, z_log_var1, logit])
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
# scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                                      reduction=tf.keras.losses.Reduction.AUTO)
def loss_fun(y, y_pred, mean_pred, log_var_pred, beta):
    '''do not consider padding'''
    # reconstruction loss
    non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(tf.cast(scce(y, y_pred), tf.float32), 
                                                                    tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
                                                        non_pad_count), axis=1))
    # recon_loss = scce(y, y_pred)
    
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + log_var_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(log_var_pred)), axis=1))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
# all_outputs = []
# inputs = ey
# [state_h, state_c] = [z1, z1]
# for _ in range(maxlen):
#     # 한 개의 time step에서 decoder 실행
#     outputs, state_h, state_c = decoder_lstm(inputs,
#                                              initial_state=[state_h, state_c])
#     logit = logit_layer(outputs)
#     inputs = embedding_layer(tf.argmax(logit, axis=-1))
#     all_outputs.append(tf.squeeze(logit))
#%% loss 
# scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                                                      reduction=tf.keras.losses.Reduction.NONE)
# def loss_mixture_fun(y, y_pred, mean_pred1, log_var_pred1, mean_pred2, log_var_pred2, pi, beta):
#     '''do not consider padding'''
#     # reconstruction loss
#     non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
#     recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(scce(y, y_pred), 
#                                                                     tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
#                                                         non_pad_count), axis=1))
    
#     # kl-divergence loss
#     term1 = tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi * M), axis=1))
#     kl1 = tf.reduce_sum(-0.5 * (1 + log_var_pred1 - tf.math.pow(mean_pred1 - prior_mu[0, :], 2) - tf.math.exp(log_var_pred1)), axis=1, keepdims=True)
#     kl2 = tf.reduce_sum(-0.5 * (1 + log_var_pred2 - tf.math.pow(mean_pred2 - prior_mu[1, :], 2) - tf.math.exp(log_var_pred2)), axis=1, keepdims=True)
#     kl_loss = term1 + tf.reduce_mean(tf.reduce_sum(tf.multiply(pi, tf.concat((kl1, kl2), axis=1)), axis=1))
    
#     return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.001):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%%
optimizer = tf.keras.optimizers.Adam(0.005)
# optimizer1 = tf.keras.optimizers.Adam(0.005)
#%% training 
epochs = 35000
# beta = 0.1
dropout_rate = 0.5
for epoch in range(1, epochs):
    beta = kl_anneal(epoch, int(epochs/2))
    if epoch % 10 == 1:
        t1 = time.time()

    idx = np.random.randint(0, len(input_text), batch_size) # sampling random batch -> stochasticity
    # input_sequence = input_text[idx][:, ::-1]
    input_sequence = input_text[idx]
    # input_sequence_dropout = input_text[idx][:, :-1]    
    output_sequence = output_text[idx]
    # label_batch = label[idx]
    
    # '''word dropout with UNK
    # -> hold PAD and UNK word embedding vector zero vector(non-trainable)'''
    # non_pad = np.sum(input_sequence != vocab.get('<PAD>'), axis=1)
    # dropout_ = [np.random.binomial(1, dropout_rate, x-2) for x in non_pad]
    # dropout_index = [d  * np.arange(1, x-1) for d, x in zip(dropout_, non_pad)]
    # for i in range(batch_size):
    #     input_sequence_dropout[i][[d for d in dropout_index[i] if d != 0]] = vocab.get('<UNK>')
        
    with tf.GradientTape(persistent=True) as tape:
         
        # get output
        # z_mean_pred1, z_log_var_pred1, z_mean_pred2, z_log_var_pred2, sequence_pred = text_vae([input_sequence, label_batch, input_sequence_dropout])
        # z_mean_pred1, z_log_var_pred1, sequence_pred = text_vae([input_sequence, input_sequence_dropout])
        z_mean_pred1, z_log_var_pred1, sequence_pred = text_vae(input_sequence)
        # pi_hat = mixprob_vae(input_sequence)
        
        # ELBO 
        # recon_loss, kl_loss, loss = loss_mixture_fun(output_sequence, sequence_pred, z_mean_pred1, z_log_var_pred1, z_mean_pred2, z_log_var_pred2, pi_hat, beta)
        recon_loss, kl_loss, loss = loss_fun(output_sequence, sequence_pred, z_mean_pred1, z_log_var_pred1, beta)
        
        # mixture probability loss
        # mix_loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.multiply(label[idx, :], pi_hat), axis=1)))
        # mix_loss_ = mix_loss / beta
        
    grad = tape.gradient(loss, text_vae.weights)
    optimizer.apply_gradients(zip(grad, text_vae.weights))
    # grad1 = tape.gradient(mix_loss_, mixprob_vae.weights)
    # optimizer1.apply_gradients(zip(grad1, mixprob_vae.weights))
    
    if epoch % 50 == 0:
        pred_id = tf.nn.top_k(sequence_pred[:5], k=10)[1].numpy()
        pred_id = [[np.random.choice(pred_id[t][k]) for k in range(maxlen-1)] for t in range(5)]
        result = [0]*5
        for i in range(5):
            result[i] = ' '.join([num_vocab.get(x) for x in pred_id[i]])
        print(result)

    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3})'.format(epoch, t2-t1))
        # print('Text VAE loss: {:.6}, Reconstruction: {:.6}, KL: {:.6}, MIX: {:.6}'.format(loss.numpy(), recon_loss.numpy(), kl_loss.numpy(), mix_loss.numpy()))
        print('Text VAE loss: {:.6}, Reconstruction: {:.6}, KL: {:.6}'.format(loss.numpy(), recon_loss.numpy(), kl_loss.numpy()))
#%%
# K.backend.clear_session()
#%% latent generation
latent_input = layers.Input((maxlen))
# latent_label = layers.Input((M))
latent_emb = embedding_layer(latent_input)
latent_h = encoder_lstm(latent_emb)

# latent_mix_prob = mix_prob_dense(latent_h)
latent_mean1 = mean_dense1(latent_h)
latent_log_var1 = log_var_dense1(latent_h)
# latent_mean2 = mean_dense2(latent_h)
# latent_log_var2 = log_var_dense2(latent_h)

# latent_prob_sampling = tf.random.categorical(latent_mix_prob, 1)
# latent_chosen_idx = tf.concat((latent_prob_sampling, tf.cast(tf.cast(tf.logical_not(tf.cast(latent_prob_sampling, tf.bool)), tf.bool), tf.int64)), axis=1)

epsilon1 = tf.random.normal((latent_dim, ))
latent_z1 = latent_mean1 + tf.math.exp(latent_log_var1 / 2) * epsilon1
# epsilon2 = tf.random.normal((latent_dim, ))
# latent_z2 = latent_mean2 + tf.math.exp(latent_log_var2 / 2) * epsilon2

# latent_z12 = tf.concat((latent_z1[:, tf.newaxis, :], latent_z2[:, tf.newaxis, :]), axis=1)
# latent_z = tf.reduce_sum(tf.multiply(tf.cast(tf.tile(latent_chosen_idx[..., tf.newaxis], (1, 1, latent_dim)), tf.float32), latent_z12), axis=1)

# latent_z12 = tf.concat((latent_z1[:, tf.newaxis, :], latent_z2[:, tf.newaxis, :]), axis=1)
# latent_z = tf.reduce_sum(tf.multiply(latent_z12, tf.tile(latent_label[..., tf.newaxis], (1, 1, latent_dim))), axis=1)

# latent_model = K.models.Model([latent_input, latent_label], latent_z)
latent_model = K.models.Model(latent_input, latent_z1)
latent_model.summary()
#%% inference model
# inf_input = layers.Input((maxlen-1))
inf_hidden = layers.Input((latent_dim))
inf_hiddens = layers.RepeatVector(maxlen-1)(inf_hidden)
# inf_emb = embedding_layer(inf_input) 
# latent_init_h = reweight_h_dense(inf_hidden)
# latent_init_c = reweight_c_dense(inf_hidden)
# inf_output = logit_layer(decoder_lstm(inf_emb, initial_state=[latent_init_h, latent_init_c]))
# inf_output = logit_layer(decoder_lstm(tf.concat((inf_emb, inf_hiddens), axis=-1)))
# inf_output = logit_layer(decoder_lstm(inf_emb, initial_state=[inf_hidden, inf_hidden]))
inf_output = logit_layer(decoder_lstm(inf_hiddens, initial_state=[inf_hidden, inf_hidden]))
inference_model = K.models.Model(inf_hidden, inf_output)
inference_model.summary()
#%% interpolation & inference
j1 = 2000 # 145
j2 = 2001 # 146
print('===input===')
print(' '.join([num_vocab.get(x) for x in input_text[j1, :] if x != 0]))
print(' '.join([num_vocab.get(x) for x in input_text[j2, :] if x != 0]))
# z1 = latent_model([input_text[[j1], :], label[[j1], :]])
# z2 = latent_model([input_text[[j2], :], label[[j2], :]])
z1 = latent_model(input_text[[j1], :])
z2 = latent_model(input_text[[j2], :])

# generating z
# z1 = np.squeeze(np.array([np.random.normal(a, 1, 1) for a in prior_mu[0, :]]))[np.newaxis, :]
# z2 = np.squeeze(np.array([np.random.normal(a, 1, 1) for a in prior_mu[1, :]]))[np.newaxis, :]

# interpolation
z_inter = z1
for v in np.linspace(0, 1, 7):
    z_inter = np.vstack((z_inter, v * z1 + (1 - v) * z2))
z_inter = np.vstack((z_inter, z2))

val_seq = np.zeros((len(z_inter), maxlen-1))
val_seq[:, 0] = vocab.get('<sos>')
result = ['']*len(z_inter)

topk = 5
for k in range(len(result)):
    for t in range(1, maxlen-1):
        pred = tf.math.softmax(inference_model([val_seq[[k], :], z_inter[[k], :]]), axis=-1)
        # pred_id = tf.argmax(pred[0][t-1]).numpy()
        '''top-k sampling'''
        pred_id = np.random.choice(np.argsort(pred[0][t-1])[::-1][:topk])
        '''neural sampling'''
        result[k] += num_vocab.get(pred_id) + ' '
        
        if num_vocab.get(pred_id) == '<eos>':
            break
    
        val_seq[:, t] = pred_id
print('===output===')
# print(result)
print('top_k: {}'.format(topk))
for i in range(len(result)):
    print(result[i])
    # print('\n')
#%% interpolation & inference
j1 = 2000 # 145
j2 = 2001 # 146
print('===input===')
print(' '.join([num_vocab.get(x) for x in input_text[j1, :] if x != 0]))
print(' '.join([num_vocab.get(x) for x in input_text[j2, :] if x != 0]))
# z1 = latent_model([input_text[[j1], :], label[[j1], :]])
# z2 = latent_model([input_text[[j2], :], label[[j2], :]])
z1 = latent_model(input_text[[j1], :])
z2 = latent_model(input_text[[j2], :])

# interpolation
z_inter = z1
for v in np.linspace(0, 1, 7):
    z_inter = np.vstack((z_inter, v * z1 + (1 - v) * z2))
z_inter = np.vstack((z_inter, z2))

infer = inference_model(z_inter)
result = ['']*len(z_inter)
topk = 1
for k in range(len(result)):
    pred = tf.math.softmax(infer[k], axis=-1)
    '''top-k sampling'''
    pred_id = [np.random.choice(x) for x in np.argsort(pred, axis=-1)[::-1][:, :topk]]
    result[k] = ' '.join([num_vocab.get(x) for x in pred_id])

print('===output===')
# print(result)
print('top_k: {}'.format(topk))
for i in range(len(result)):
    print(result[i])
    # print('\n')
#%%





























































