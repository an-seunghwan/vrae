#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:08:37 2020

@author: anseunghwan
"""

#%%
'''
- 수치 정보만을 이용하여 template classification
'''
#%%
import tensorflow as tf
# import tensorflow_probability as tfp
import tensorflow.keras as K
from tensorflow.keras import layers
# from tensorflow.keras import preprocessing
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
# import re
# import matplotlib.pyplot as plt
# from pprint import pprint
# from konlpy.tag import Okt
# okt = Okt()
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
'''
data loading
'''
'''barycenter(template) data'''
body = pd.read_csv('./data/body_template.csv')
body.head()
body.columns
semantic = body[['sentence', '시맨틱']]
semantic_dict = {x:y for x,y in zip(semantic['sentence'], semantic['시맨틱'])}
# 중복된 행 제거
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)

# template
template = body.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)

template_dict = {x:i for i,x in enumerate(sorted(uniqe_template))}
template_num_dict = {y:x for x,y in template_dict.items()}

'''user id list'''
user_list_total = sorted(list(set(body['USER_ID'].to_list())))
len(user_list_total)
np.random.seed(1)
idx = np.random.choice(range(len(user_list_total)), int(len(user_list_total) * 0.8), replace=False)
user_list = list(np.array(user_list_total)[idx])
user_val_list = [x for x in user_list_total if x not in user_list]

'''sas dict data'''

with open("./body_final/sas_dict_scaled.pkl", "rb") as f:
    sas_dict = pickle.load(f)
    
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
# data_num = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '12']
data_num = ['00', '02', '03', '04', '05', '06', '09', '12']
datamart_dict = {}
for n, num in enumerate(data_num):
    datamart = pd.read_csv('./body_final/body_datamart_V1_{}.csv'.format(num), skiprows=0).iloc[:-2]
    dict_ = {x:y for x,y in zip(datamart['변수 이름'], datamart['레이블'])}
    # datamart_dict[n] = dict_
    datamart_dict.update(dict_)

# custom defined variable
datamart_dict['BLOOD_PRESS_L'] = '최저혈압'
datamart_dict['BLOOD_PRESS_U'] = '최고혈압'
datamart_dict['TARGET_HEART_L'] = '최저목표심박수'
datamart_dict['TARGET_HEART_U'] = '최고목표심박수'
#%%
'''
- user exercise data generator
- sentence가 없는 exercise data가 존재
- 변수 filtering 필요
'''

'''
- filtering 변수
- 숫자 정보만을 이용!
'''
var00 = ['PHSC_LVL', 'DAY_ACT_CAL', 'DAY_ACT_TM', 'ACT_VALID_LIMIT', 'ACT_SAFETY_LIMIT', 'WALK_CNT',
         'MAX_OXY_INTAKE_AM', 'HR_CLF', 'CALM_HR', 'VALID_OBJ_HR', 'SAFETY_OBJ_HR']
var02 = ['HEIGHT', 'WEIGHT', 'BMI', 'BODY_FAT_PER', 'BLOOD_PRESS_L', 'BLOOD_PRESS_U', 'BLOOD_SUGAR', 'HDL_CHOL', 'NEUTRAL_FAT',
         'WAIST_MSMT', 'MUSCLE', 'BONE_MUSCLE']
var03 = ['RECOM_CAL', 'TARGET_HEART_L', 'TARGET_HEART_U', 'DAY_RECOM_CAL', 'DAY_RECOM_TIME', 'TARGET_EX_TIME', 
         'ACT_VALID_LIM', 'ACT_SAFE_LIM', 'TARGET_WALK']
var04 = ['DAY_ENG_NEED_AM', 'GR_INTAKE_CNT', 'MT_INTAKE_CNT', 'VG_INTAKE_CNT', 'FR_INTAKE_CNT', 'MK_INTAKE_CNT', 'WEIGHT_04', 'OBJ_WEIGHT']
var05 = ['DAY_AVE_ACT_1', 'DAY_AVE_ACT_2', 'DAY_AVE_ACT_3', 'DAY_AVE_ACT_4', 'DAY_AVE_ACT_5', 'DAY_AVE_ACT_6', 'DAY_AVE_ACT_7']
var06 = ['AVE_WALK_N', 'OBJ_WALK_DAY', 'OBJ_WALK_RATE']
var09 = ['WALK_CNT_09', 'MOD_EXE_TM_09', 'CONSUME_CAL_09', 'MOVE_DIST_09', 'AVE_WALK_CNT_09', 'AVE_MOD_EXE_TM_09',
         'AVE_CONSUME_CAL_09', 'AVE_MOVE_DIST_09', 'ACT_DAYS_09']
var12 = ['TOT_EXCS_TM', 'EXCS_TM', 'EXCS_RATE']
var_filter = ['USER_ID', 'CNSL_SN', 'sentence'] + var00 + var02 + var03 + var04 + var05 + var06 + var09 + var12

# what is filtered?
# var_filter_kor = [(x, y) for x,y in datamart_dict.items() if x in var_filter]
# var_filter_kor

data_num = ['00', '02', '03', '04', '05', '06', '09', '12']

# def user_df_generator():
#     i = 0
#     while i < len(user_list):
#         # exercise data
#         user_key = user_list[i]
#         for n, num in enumerate(data_num):
#             if n == 0:
#                 user_df = sas_dict[n].loc[sas_dict[n]['USER_ID_N'] == user_key].reset_index(drop=True)
#             else:
#                 user_df = pd.merge(user_df, sas_dict[n].loc[sas_dict[n]['USER_ID_N'] == user_key].reset_index(drop=True), how='outer', on='CNSL_SN_N')
#         # sentence data
#         user_df = pd.merge(body.loc[body['USER_ID'] == user_key].reset_index(drop=True), user_df, how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
#         user_df = user_df[var_filter]

#         # missing values to 0 (None, nan)
#         user_df = user_df.fillna(-1)
#         i += 1
#         yield user_df

def user_df_batch_generator(batch_size):
    i = 0
    while True:
        # exercise data
        idx = np.random.randint(0, len(user_list), size=batch_size)
        user_key = list(np.array(user_list)[idx])
        for n, num in enumerate(data_num):
            if n == 0:
                user_df = sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True)
            else:
                user_df = pd.merge(user_df, 
                                   sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True), 
                                   how='outer', on='CNSL_SN_N')
        
        # sentence data
        user_df = pd.merge(body.loc[np.isin(body['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                           user_df, 
                           how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]
        
        # missing values to 0 (None, nan)
        user_df = user_df.fillna(-1)
        
        '''reshape to matrix: rows for each sas data'''
        for n, num in enumerate(data_num):
            if n == 0:
                temp = user_df[globals()['var{}'.format(num)]].to_numpy()
                user_x = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
            else:
                temp = user_df[globals()['var{}'.format(num)]].to_numpy()
                temp = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
                user_x = np.append(user_x, temp, axis=1)
        user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        
        i += 1
        yield user_x, user_y
#%%
'''
- user df 내에서도 sentence의 중복이 발생 (일부의 exercise data만 다르기 때문에 문제 발생)
'''
batch_size = 10
max_var_num = max(len(globals()['var{}'.format(num)]) for num in data_num)
user_df_batch_gen = user_df_batch_generator(batch_size)
user_x, user_y = next(user_df_batch_gen)
user_y = np.array([template_dict.get(x) for x in user_y.to_numpy()[:, 2]])[:, np.newaxis]
#%% input
var_num = user_x.shape[-1]
target_num = len(template_dict)
sas_num = len(data_num)
latent_dim = 10
#%% modeling
'''encoder with attention'''
input_layer = layers.Input((sas_num, var_num))
w1 = layers.Dense(1, use_bias=False, activation='tanh')
attention = tf.nn.softmax((w1(input_layer)), axis=1)
h = tf.squeeze(tf.matmul(attention, input_layer, transpose_a=True), axis=1)

w2 = layers.Dense(latent_dim)
h = w2(h)

mean_dense = layers.Dense(latent_dim)
log_var_dense = layers.Dense(latent_dim, activation='tanh')
z_mean = mean_dense(h)
z_log_var = log_var_dense(h)
epsilon = tf.random.normal((latent_dim, ))
z = z_mean + tf.math.exp(z_log_var / 2) * epsilon

'''decoder'''
# reconstruction y
w3 = layers.Dense(100)
y_recon_layer = layers.Dense(target_num)
h = w3(layers.concatenate((tf.reshape(input_layer, (-1, sas_num*var_num)), z)))
y_recon = y_recon_layer(h)

vrae = K.models.Model(input_layer, [z_mean, z_log_var, z, y_recon, attention])
vrae.summary()  
#%%
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)

def loss_fun(y, y_pred, mean_pred, log_var_pred, beta):
    # '''do not consider padding'''
    # reconstruction loss
    # non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    # recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(tf.cast(scce(y, y_pred), tf.float32), 
    #                                                                 tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
    #                                                     non_pad_count), axis=1))
    recon_loss = tf.reduce_mean(scce(y, y_pred), keepdims=False)
    
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + log_var_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(log_var_pred)), axis=1))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.001):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%%
optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
#%%
# for iteration in range(10):
#     user_df_gen = user_df_generator()
epochs = 8500

'''python generator -> tf data pipeline (padding is needed)'''
user_df_batch_gen = user_df_batch_generator(batch_size)
# dataset = tf.data.Dataset.from_generator(user_df_batch_generator, tf.float32)
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()

for epoch in range(1, epochs):
    
    beta = kl_anneal(epoch, int(epochs/2))
    if epoch % 10 == 1:
        t1 = time.time()

    x_data, y_data = next(user_df_batch_gen)
    x_data = tf.convert_to_tensor(x_data.astype(np.float32), dtype=tf.float32)
    y_data = np.array([template_dict.get(x) for x in y_data.to_numpy()[:, 2]])[:, np.newaxis]
    y_data = tf.convert_to_tensor(y_data, dtype=tf.int32)
    
    with tf.GradientTape(persistent=True) as tape:
        
        mean, logvar, z_pred, y_pred, attention = vrae(x_data)
        
        recon_loss, kl_loss, loss = loss_fun(y_data, y_pred, mean, logvar, beta)
        
    grad = tape.gradient(loss, vrae.weights)
    optimizer.apply_gradients(zip(grad, vrae.weights))
    
    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3}) VRAE loss: {:.6}'.format(epoch, t2-t1, loss.numpy()))
        print('Y RECON loss: {:.6}, KL loss: {:.6}'.format(recon_loss.numpy(), kl_loss.numpy()))
#%%
# K.backend.clear_session()
#%% inference
latent_input = layers.Input((sas_num, var_num))
latent_attention = tf.nn.softmax((w1(latent_input)), axis=1)
latent_h = tf.squeeze(tf.matmul(latent_attention, latent_input, transpose_a=True), axis=1)
latent_h = w2(latent_h)

latent_mean = mean_dense(latent_h)
latent_log_var = log_var_dense(latent_h)
epsilon = tf.random.normal((latent_dim, ))
latent = latent_mean + tf.math.exp(latent_log_var / 2) * epsilon

h = w3(layers.concatenate((tf.reshape(latent_input, (-1, sas_num*var_num)), latent)))
y_hat = y_recon_layer(h)

latent_model = K.models.Model(latent_input, latent)
latent_model.summary()

inference_vrae = K.models.Model(latent_input, [latent_mean, latent_log_var, latent, y_hat, latent_attention])
inference_vrae.summary()  
#%%
def user_df_generator_val():
    for i in range(len(user_val_list)):
        # exercise data
        user_key = user_val_list[i]
        for n, num in enumerate(data_num):
            if n == 0:
                user_df = sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True)
            else:
                user_df = pd.merge(user_df, 
                                   sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True), 
                                   how='outer', on='CNSL_SN_N')
        
        # sentence data
        user_df = pd.merge(body.loc[np.isin(body['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                           user_df, 
                           how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]
        
        # missing values to 0 (None, nan)
        user_df = user_df.fillna(-1)
        
        '''reshape to matrix: rows for each sas data'''
        for n, num in enumerate(data_num):
            if n == 0:
                temp = user_df[globals()['var{}'.format(num)]].to_numpy()
                user_x = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
            else:
                temp = user_df[globals()['var{}'.format(num)]].to_numpy()
                temp = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
                user_x = np.append(user_x, temp, axis=1)
        user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        
        yield user_x, user_y
#%% interpolation & inference
user_df_gen_val = user_df_generator_val()
count = 0
n = 0
attention_result = []
pred_result = []
topk = 5
for i in tqdm(range(len(user_val_list))):
    x_data, y_data = next(user_df_gen_val)
    x_data = tf.convert_to_tensor(x_data.astype(np.float32), dtype=tf.float32)
    y_data = np.array([template_dict.get(x) for x in y_data.to_numpy()[:, 2]])[:, np.newaxis]
    y_data = tf.convert_to_tensor(y_data, dtype=tf.int32)
    
    _, _, _, y_inf, attention_inf = inference_vrae(x_data)
    
    # argmax
    # count += sum(np.squeeze(y_data) == np.argmax(y_inf, axis=1))
    # pred_result.append(np.argmax(y_inf, axis=1))
    
    # topk
    count += sum(np.isin(y_data.numpy(), np.argsort(y_inf, axis=1)[:, -topk:][:, ::-1]))
    pred_result.append(np.argsort(y_inf, axis=1)[:, -topk:][:, ::-1])
    
    n += len(y_data)
    attention_result.append(attention_inf.numpy())
    
print(count / n)
#%%
def specific_user(user_key):
    # exercise data
    for n, num in enumerate(data_num):
        if n == 0:
            user_df = sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True)
        else:
            user_df = pd.merge(user_df, 
                               sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True), 
                               how='outer', on='CNSL_SN_N')
    
    # sentence data
    user_df = pd.merge(body.loc[np.isin(body['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                       user_df, 
                       how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
    user_df = user_df[var_filter]
    
    # missing values to 0 (None, nan)
    user_df = user_df.fillna(-1)
    
    '''reshape to matrix: rows for each sas data'''
    for n, num in enumerate(data_num):
        if n == 0:
            temp = user_df[globals()['var{}'.format(num)]].to_numpy()
            user_x = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
        else:
            temp = user_df[globals()['var{}'.format(num)]].to_numpy()
            temp = np.append(temp, np.zeros((temp.shape[0], max_var_num-temp.shape[1])), axis=1)[:, np.newaxis, :]
            user_x = np.append(user_x, temp, axis=1)
    user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
    
    return user_x, user_y
#%% show results
attention_sentence = []
for j in tqdm(range(len(user_val_list))):
    user_key = user_val_list[j]
    _, uy = specific_user(user_key)
    sas_argmax = np.squeeze(np.argmax(attention_result[j], axis=1), axis=-1)
    attention_var = [globals()['var{}'.format(data_num[n])] for n in sas_argmax]
    pred = pred_result[j]
    
    for k in range(len(attention_var)):
        a = sas_argmax[k]
        b = ' + '.join([datamart_dict.get(x) for x in attention_var[k]])
        c = uy['sentence'].iloc[k].replace(u'\xa0', '')
        c = '({}):{}'.format(semantic_dict.get(c), c)
        d = [template_num_dict.get(p).replace(u'\xa0', '') for p in pred[k]]
        d = ['({}):{}'.format(semantic_dict.get(x), x) for x in d]
        attention_sentence.append([a, b, c] + d)

# temp = [[a, b, c.replace(u'\xa0', ''), d.replace(u'\xa0', '')] for a,b,c,d in attention_sentence]
with open('./result/attention_sentence_topk.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['attention sas number', 'attention variable', 'true'] + ['predict{}'.format(i) for i in range(topk)])
    for k in range(len(attention_sentence)): 
        wr.writerow(attention_sentence[k])
#%%




































































