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
import pickle
import sklearn
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
# 중복된 행 제거
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)

# template
template = body.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)

template_dict = {x:i for i,x in enumerate(sorted(uniqe_template))}

'''user id list'''
user_list = sorted(list(set(body['USER_ID'].to_list())))
len(user_list)

'''sas dict data'''

with open("./body_final/sas_dict.pkl", "rb") as f:
    sas_dict = pickle.load(f)
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

def user_df_generator():
    i = 0
    while i < len(user_list):
        # exercise data
        user_key = user_list[i]
        for n, num in enumerate(data_num):
            if n == 0:
                user_df = sas_dict[n].loc[sas_dict[n]['USER_ID_N'] == user_key].reset_index(drop=True)
            else:
                user_df = pd.merge(user_df, sas_dict[n].loc[sas_dict[n]['USER_ID_N'] == user_key].reset_index(drop=True), how='outer', on='CNSL_SN_N')
        # sentence data
        user_df = pd.merge(body.loc[body['USER_ID'] == user_key].reset_index(drop=True), user_df, how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]

        # missing values to 0 (None, nan)
        user_df = user_df.fillna(-1)
        i += 1
        yield user_df
#%%
'''
- user df 내에서도 sentence의 중복이 발생 (일부의 exercise data만 다르기 때문에 문제 발생)
'''
user_df_gen = user_df_generator()
user_df = next(user_df_gen)
x_data = user_df.to_numpy()[:, 3:]
x_data.shape
y_data = np.array([template_dict.get(x) for x in user_df.to_numpy()[:, 2]])[:, np.newaxis]
y_data.shape
#%% input
var_num = x_data.shape[1]
target_num = len(template_dict)
sas_num = len(data_num)
latent_dim = 10
#%% modeling
'''encoder'''
input_layer = layers.Input((var_num))
x = layers.RepeatVector(sas_num)(input_layer)
w1 = layers.Dense(1, use_bias=False, activation='tanh')
attention = tf.nn.softmax(w1(x), axis=1)
h = tf.squeeze(tf.matmul(attention, x, transpose_a=True), axis=1)
w2 = layers.Dense(latent_dim)
h = w2(h)

mean_dense = layers.Dense(latent_dim)
log_var_dense = layers.Dense(latent_dim, activation='tanh')
z_mean = mean_dense(h)
z_log_var = log_var_dense(h)
epsilon = tf.random.normal((latent_dim, ))
z = z_mean + tf.math.exp(z_log_var / 2) * epsilon

encoder = K.models.Model(input_layer, z)
encoder.summary()  

'''decoder'''
# reconstruction x
x_recon_layer = layers.Dense(var_num)
x_recon = x_recon_layer(z)

# reconstruction y
y_recon_layer = layers.Dense(target_num)
y_recon = y_recon_layer(z)
#%%
def build_prior_discriminator():
    z_input = layers.Input((latent_dim))
    
    z = layers.LeakyReLU(alpha=0.2)(z_input)
    z = layers.Dense(1, activation='sigmoid')(z)
    
    return K.models.Model(z_input, z)
#%%
D_prior = build_prior_discriminator()
D_prior.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.05),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
#%%
'''model'''
D_prior.trainable = False
validity_x = D_prior(z)
vrae = K.models.Model(input_layer, [z_mean, z_log_var, z, validity_x, x_recon, y_recon])
vrae.summary()  
#%%
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)
mse = tf.keras.losses.MeanSquaredError()
def loss_fun(x, x_pred, y, y_pred, beta):
    # x reconstruction loss
    x_recon_loss = mse(x, x_pred)
    
    '''do not consider padding'''
        # y reconstruction loss
    non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    y_recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(tf.cast(scce(y, y_pred), tf.float32), 
                                                                      tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
                                                          non_pad_count), axis=1))
    
    return x_recon_loss + beta * y_recon_loss, x_recon_loss, y_recon_loss
#%%
batch_size = 500
epochs = len(user_list)
#%%
optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
#%%
for iteration in range(10):
    user_df_gen = user_df_generator()
    for epoch in range(epochs):
    
        user_df = next(user_df_gen)
        # batch normalization
        x_data = user_df.to_numpy()[:, 3:].astype(np.float32)
        # sklearn.preprocessing.scale(x_data)
        y_data = np.array([template_dict.get(x) for x in user_df.to_numpy()[:, 2]])[:, np.newaxis]
        
        valid = np.ones((len(y_data), 1))
        fake = np.ones((len(y_data), 1))
        
        # GAN
        latent_real = np.random.normal(size=(len(y_data), latent_dim))
        d_loss_real = D_prior.train_on_batch(latent_real, valid)
        latent_fake = encoder(x_data)
        d_loss_fake = D_prior.train_on_batch(latent_fake, fake)
        d_loss_prior = np.add(d_loss_real, d_loss_fake) / 2
        
        with tf.GradientTape(persistent=True) as tape:
            
            mean, logvar, latent, vx, x_pred, y_pred = vrae(x_data)
            
            recon_loss, x_recon_loss, y_recon_loss = loss_fun(x_data, x_pred, y_data, y_pred, beta=1)
            
            valid_loss = tf.cast(tf.reduce_mean(tf.keras.losses.binary_crossentropy(vx, valid), keepdims=True), tf.float32)
            
            loss = recon_loss + 0.1 * valid_loss
        
        grad = tape.gradient(loss, vrae.weights)
        optimizer.apply_gradients(zip(grad, vrae.weights))
        
        if epoch % 10 == 0:
            print('({} iteration {} epoch) VRAE loss: {:.6}'.format(iteration, epoch, loss.numpy()[0]))
            print('X RECON loss: {:.6}, Y RECON loss: {:.6}, VALID loss: {:.6}'.format(x_recon_loss.numpy(), y_recon_loss.numpy(), valid_loss.numpy()[0]))
#%%
# K.backend.clear_session()
#%%








































































