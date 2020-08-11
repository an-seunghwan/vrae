#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:41:03 2020

@author: anseunghwan
"""

#%%
'''
- 수치 정보만을 이용하여 template classification
- annotation matrix
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

semantic_data = body[['sentence', '시맨틱']]
semantic_id = {x:i for i,x in enumerate(['경고', '권장', '지식', '현상'])}
semantic_numid = {y:x for x,y in semantic_id.items()}
semantic_dict = {x:semantic_id.get(y) for x,y in zip(semantic_data['sentence'], semantic_data['시맨틱'])}

# 중복된 행 제거
template_data = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
len(template_data)

# template dictionary for each semantic
template = body.sentence
unique_template = list(set(template.to_list()))
len(unique_template)

semantic_template_dict = {}
for i in range(len(semantic_id)):
    temp = sorted([x for x in unique_template if semantic_dict.get(x) == i])
    semantic_template_dict[i] = {x:i+1 for i,x in enumerate(temp)}
    semantic_template_dict[i][0] = '<NONE>'

semantic_template_numdict = {}
for i in range(len(semantic_template_dict)):
    temp = semantic_template_dict.get(i)
    semantic_template_numdict[i] = {y:x for x,y in temp.items()}
#%%
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
max_var_num = max(len(globals()['var{}'.format(num)]) for num in data_num)

def user_df_batch_generator(batch_size):
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
        
        # missing values (None, nan) to 0 
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
                
        '''y label'''
        # user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        user_y = user_df['sentence']
        temp = [semantic_dict.get(x) for x in user_y.to_numpy()]
        label_y = np.zeros((user_y.shape[0], len(semantic_id)))
        for i in range(len(temp)):
            label_y[i, temp[i]] = semantic_template_dict.get(temp[i]).get(user_y.to_numpy()[i])
        
        yield user_x, label_y
#%%
'''
- user df 내에서도 sentence의 중복이 발생 (일부의 exercise data만 다르기 때문에 문제 발생)
'''
batch_size = 10
user_df_batch_gen = user_df_batch_generator(batch_size)
user_x, label_y = next(user_df_batch_gen)
user_x.shape
label_y.shape
#%% input
target_num = [len(x) for x in semantic_template_dict.values()]
sas_num = len(data_num)
latent_dim = 10
#%% modeling
'''encoder with attention using for loop'''
input_layer = layers.Input((sas_num, max_var_num))
w1 = [layers.Dense(1, use_bias=False, activation='tanh') for _ in range(len(semantic_id))]
attention = [tf.nn.softmax((w(input_layer)), axis=1) for w in w1]
h = [tf.squeeze(tf.matmul(a, input_layer, transpose_a=True), axis=1) for a in attention]

mean_dense = [layers.Dense(latent_dim) for _ in range(len(semantic_id))]
logvar_dense = [layers.Dense(latent_dim, activation='tanh') for _ in range(len(semantic_id))]
z_mean = [d(x) for d,x in zip(mean_dense, h)]
z_logvar = [d(x) for d,x in zip(logvar_dense, h)]
epsilon = [tf.random.normal((latent_dim, )) for _ in range(len(semantic_id))]
z = [m + tf.math.exp(v / 2) * e for m,v,e in zip(z_mean, z_logvar, epsilon)]

'''decoder'''
# reconstruction y
w3 = [layers.Dense(100) for _ in range(len(semantic_id))]
y_recon_layer = [layers.Dense(n) for n in target_num]
h = [layers.concatenate((tf.reshape(input_layer, (-1, sas_num*max_var_num)), x)) for x in z]
h = [w(x) for w,x in zip(w3, h)]
y_recon = [d(x) for d,x in zip(y_recon_layer, h)]

for i in range(len(attention)):
    if i == 0:
        attention_matrix = attention[i]
    else:
        attention_matrix = layers.concatenate((attention_matrix, attention[i]), axis=-1)

annotation_model = K.models.Model(input_layer, attention_matrix)
vrae = K.models.Model(input_layer, [z_mean, z_logvar, z, y_recon, attention_matrix])
vrae.summary()  
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.0005):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%%
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)

def loss_fun(y, y_pred, mean, logvar, beta):
    # '''do not consider NONE'''
    # reconstruction loss
    recon_loss = tf.zeros(())
    for i in range(len(semantic_id)):
        recon_loss += tf.divide(tf.reduce_mean(tf.multiply(scce(y[:, i], y_pred[i]), tf.cast(y[:, i] != 0, tf.float32)), keepdims=True), len(semantic_id))
    
    # kl-divergence loss
    kl_loss = tf.zeros(())
    for i in range(len(semantic_id)):
        kl_loss += tf.divide(tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar_pred[i] - tf.math.pow(mean_pred[i], 2) - tf.math.exp(logvar_pred[i])), axis=1), 
                                            keepdims=True),
                             len(semantic_id))
    
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
optimizer1 = tf.keras.optimizers.Adam(0.0002, 0.5)
#%%
epochs = 7270

'''python generator -> tf data pipeline (padding is needed)'''
user_df_batch_gen = user_df_batch_generator(batch_size)
# dataset = tf.data.Dataset.from_generator(user_df_batch_generator, tf.float32)
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()

for epoch in range(1, epochs):
    
    beta = kl_anneal(epoch, int(epochs/2))
    if epoch % 10 == 1:
        t1 = time.time()

    x_data, label_y = next(user_df_batch_gen)
    x_data = tf.convert_to_tensor(x_data.astype(np.float32), dtype=tf.float32)
    label_y = tf.convert_to_tensor(label_y, dtype=tf.int32)
    
    with tf.GradientTape(persistent=True) as tape:
        
        mean_pred, logvar_pred, z_pred, y_pred, annotation = vrae(x_data)
        
        recon_loss, kl_loss, loss = loss_fun(label_y, y_pred, mean_pred, logvar_pred, beta)
        
        aa = tf.matmul(annotation, annotation, transpose_a=True) - tf.cast(tf.linalg.tensor_diag(np.ones(len(semantic_id))), tf.float32)
        aa_loss = 0.2*tf.reduce_mean(tf.linalg.norm(aa, axis=[1, 2]), keepdims=True)
        
    grad = tape.gradient(loss, vrae.weights)
    optimizer.apply_gradients(zip(grad, vrae.weights))
    grad1 = tape.gradient(aa_loss, annotation_model.weights)
    optimizer1.apply_gradients(zip(grad1, annotation_model.weights))
    
    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3}) VRAE loss: {:.6}'.format(epoch, t2-t1, loss.numpy()[0]))
        print('Y RECON loss: {:.6}, KL loss: {:.6}, Annotation loss: {:.6}'.format(recon_loss.numpy()[0], kl_loss.numpy()[0], aa_loss.numpy()[0]))
#%%
# K.backend.clear_session()
#%% inference
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
        
        '''y label'''
        user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        # user_y = user_df['sentence']
        temp = [semantic_dict.get(x) for x in user_y['sentence'].to_numpy()]
        label_y = np.zeros((user_y.shape[0], len(semantic_id)))
        for i in range(len(temp)):
            label_y[i, temp[i]] = semantic_template_dict.get(temp[i]).get(user_y['sentence'].to_numpy()[i])
                
        yield user_x, user_y, label_y
#%% validation error and show result
'''attention result는 나중에!'''
user_df_gen_val = user_df_generator_val()
count_argmax = 0
count_topk = 0
n = 0
topk = 5
val_result = {'id':['id'], 'true':['true'], 0:['경고'], 1:['권장'], 2:['지식'], 3:['현상']}
for i in tqdm(range(len(user_val_list))):
    x_data, user_y, label_y = next(user_df_gen_val)
    x_data = tf.convert_to_tensor(x_data.astype(np.float32), dtype=tf.float32)
    label_y = tf.convert_to_tensor(label_y, dtype=tf.int32)
    
    _, _, _, y_inf, _ = vrae(x_data)
    
    y_argmax = []
    for i in range(len(semantic_id)):
        y_argmax.append(list(np.argmax(y_inf[i], axis=1)))
        
    for k in range(len(user_y)):
        answer = user_y['sentence'].iloc[k].replace(u'\xa0', '')
        answer = '({}):{}'.format(semantic_numid.get(semantic_dict.get(answer)), answer)
        val_result['true'] = val_result.get('true') + [answer]
        
        user_id = '_'.join([str(x) for x in user_y[['USER_ID', 'CNSL_SN']].iloc[0].to_list()])
        val_result['id'] = val_result.get('id') + [user_id]
        
        for i in range(len(semantic_id)):
            pred_ = '({}):{}'.format(semantic_numid.get(semantic_dict.get(semantic_template_numdict[i].get(y_argmax[i][k]))), 
                                     semantic_template_numdict[i].get(y_argmax[i][k])).replace(u'\xa0', '')
            
            val_result[i] = val_result.get(i) + [pred_]
    
    idx_ = np.where(label_y.numpy()[0, :] != 0)[0][0]
    
    # argmax
    count_argmax += sum(label_y.numpy()[:, idx_] == np.argmax(y_inf[idx_], axis=1))      
    
    # topk
    count_topk += sum(np.isin(label_y.numpy()[:, idx_], np.argsort(y_inf[idx_], axis=1)[:, -topk:][:, ::-1]))
    
    n += len(label_y)
    
print(count_argmax / n)
print(count_topk / n)

with open('./result/validation_guidance_200810.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    for k in range(len(val_result[0])): 
        wr.writerow([val_result['id'][k], val_result['true'][k], val_result[0][k], val_result[1][k], val_result[2][k], val_result[3][k]])
#%% show results
# id_answer = []
# attention_template = {-1:['true'], 0:['경고'], 1:['권장'], 2:['지식'], 3:['현상']}
# for j in tqdm(range(len(user_val_list))):
#     user_key = user_val_list[j]
#     ux, uy, ly = specific_user(user_key)
#     _, _, _, y_inf, attention_inf = vrae(ux)
    
#     sas_argmax = np.argmax(attention_inf.numpy(), axis=1)   
#     attention_var = [[globals()['var{}'.format(data_num[n])] for n in sas_argmax[:, i]] for i in range(len(semantic_id))]
    
#     y_argmax = []
#     for i in range(len(semantic_id)):
#         y_argmax.append(list(np.argmax(y_inf[i], axis=1)))
        
#     for k in range(len(uy)):
#         answer = uy['sentence'].iloc[k].replace(u'\xa0', '')
#         answer = '({}):{}'.format(semantic_numid.get(semantic_dict.get(answer)), answer)
#         for i in range(len(semantic_id)):
#             a = ' + '.join([datamart_dict.get(x) for x in attention_var[i][k]])    
#             b = '({}):{}'.format(semantic_numid.get(semantic_dict.get(semantic_template_numdict[i].get(y_argmax[i][k]))), 
#                                  semantic_template_numdict[i].get(y_argmax[i][k])).replace(u'\xa0', '')
            
#             user_id = '_'.join([str(x) for x in uy[['USER_ID', 'CNSL_SN']].iloc[k].to_list()])
                        
#             attention_template[i] = attention_template.get(i) + [[a], [b]]
#             id_answer.extend([[user_id], [answer]])

# # temp = [[a, b, c.replace(u'\xa0', ''), d.replace(u'\xa0', '')] for a,b,c,d in attention_sentence]
# attention_template_ = {x:y[1:] for x,y in attention_template.items()}
# with open('./result/validation_200810.csv', 'w', encoding='euc-kr', newline='') as f:
#     wr = csv.writer(f)
#     wr.writerow(['answer'] + list(semantic_id.keys()))
#     for k in range(len(attention_template_[0])): 
#         wr.writerow(id_answer[k] + attention_template_[0][k] + attention_template_[1][k] + attention_template_[2][k] + attention_template_[3][k])
#%%






































