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
- model tunning
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
import math
import csv    
import time
import pickle
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
health = pd.read_csv('./data/barycenter_health.csv')
health.head()
health.columns

semantic_data = health[['sentence', '시맨틱']]
semantic_id = {x:i for i,x in enumerate(['경고', '권장', '지식', '현상'])}
semantic_numid = {y:x for x,y in semantic_id.items()}
semantic_dict = {x:semantic_id.get(y) for x,y in zip(semantic_data['sentence'], semantic_data['시맨틱'])}

# 중복된 행 제거
template_data = health[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
len(template_data)

# template dictionary for each semantic
template = health.sentence
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
user_list_total = sorted(list(set(health['USER_ID'].to_list())))
len(user_list_total)
np.random.seed(1)
idx = np.random.choice(range(len(user_list_total)), int(len(user_list_total) * 0.9), replace=False)
user_list = list(np.array(user_list_total)[idx])
user_val_list = [x for x in user_list_total if x not in user_list]

'''sas dict data'''

with open("./health_final/sas_dict_scaled.pkl", "rb") as f:
    sas_dict = pickle.load(f)
    
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
data_num = ['02', '03', '04', '07', '08', '09', '10',
            '11', '12', '13', '14', '15', '16']
datamart_dict = {}
for n, num in enumerate(data_num):
    datamart = pd.read_csv('./health_final/health_datamart_V1_{}.csv'.format(num), skiprows=0).iloc[:-2]
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
var02 = ['HEIGHT', 'WEIGHT', 'BMI', 'BODY_FAT_PER', 'BLOOD_PRESS_L', 'BLOOD_PRESS_U', 'BLOOD_SUGAR', 'HDL_CHOL', 'NEUTRAL_FAT',
         'WAIST_MSMT', 'MUSCLE', 'BONE_MUSCLE']
var03 = ['RECOM_CAL', 'TARGET_HEART_L', 'TARGET_HEART_U', 'DAY_RECOM_CAL', 'DAY_RECOM_TIME', 'TARGET_EX_TIME', 
         'ACT_VALID_LIM', 'ACT_SAFE_LIM', 'TARGET_WALK']
var04 = ['DAY_ENG_NEED_AM', 'GR_INTAKE_CNT', 'MT_INTAKE_CNT', 'VG_INTAKE_CNT', 'FR_INTAKE_CNT', 'MK_INTAKE_CNT', 
         'WEIGHT_04', 'OBJ_WEIGHT_x']
# var06 = ['DAY_AVE_ACT_1', 'DAY_AVE_EXCS_1', 'DAY_AVE_ACT_2', 'DAY_AVE_EXCS_2', 'DAY_AVE_ACT_3', 'DAY_AVE_EXCS_3', 'DAY_AVE_ACT_4',
#          'DAY_AVE_EXCS_4', 'DAY_AVE_ACT_5', 'DAY_AVE_EXCS_5', 'DAY_AVE_ACT_6', 'DAY_AVE_EXCS_6', 'DAY_AVE_ACT_7', 'DAY_AVE_EXCS_7']
var07 = ['AVE_WALK_N', 'OBJ_WALK_TOT', 'OBJ_WALK_DAY']
var08 = ['WEIGHT_08']
var09 = ['OBJ_WEIGHT_y', 'CUR_WEIGHT', 'DIF_WEIGHT']
var10 = ['DAY_ENG_NEED_AM_1', 'CONSUME_CAL_1', 'DAY_ENG_NEED_AM_2', 'CONSUME_CAL_2', 'DAY_ENG_NEED_AM_3', 'CONSUME_CAL_3',
         'DAY_ENG_NEED_AM_4', 'CONSUME_CAL_4', 'DAY_ENG_NEED_AM_5', 'CONSUME_CAL_5', 'DAY_ENG_NEED_AM_6', 'CONSUME_CAL_6', 'DAY_ENG_NEED_AM_7', 'CONSUME_CAL_7']
var11 = ['MEAL_NO', 'TOT_NO', 'AVE_CAL', 'MORNING_CAL', 'LUNCH_CAL', 'DINNER_CAL', 'MORNING_MID_CAL', 'LUNCH_MID_CAL', 'DINNER_MID_CAL']
var12 = ['WEIGHT_12', 'MUSCLE_12', 'BODY_FAT_12', 'BMI_12']
var13 = ['BLOOD_PRESS_MAX', 'BLOOD_PRESS_MIN', 'PULSE']
var14 = ['BLOOD_SUGAR_14']
var15 = ['FOUR_WEEK_AVE_MAX', 'FOUR_WEEK_AVE_MIN', 'EIGHT_WEEK_AVE_MAX', 'EIGHT_WEEK_AVE_MIN', 'TWELVE_WEEK_AVE_MAX', 'TWELVE_WEEK_AVE_MIN']
var16 = ['SUGAR_FOUR_WEEK_AVE', 'SUGAR_EIGHT_WEEK_AVE', 'SUGAR_TWELVE_WEEK_AVE']
var_filter = ['USER_ID_x', 'CNSL_SN_x', 'sentence'] + var02 + var03 + var04 + var07 + var08 + var09 + \
    var10 + var11 + var12 + var13 + var14 + var15 + var16
var_numeric = var02 + var03 + var04 + var07 + var08 + var09 + \
    var10 + var11 + var12 + var13 + var14 + var15 + var16
# what is filtered?
# var_filter_kor = [(x, y) for x,y in datamart_dict.items() if x in var_filter]
# var_filter_kor

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
        user_df = pd.merge(health.loc[np.isin(health['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                           user_df, 
                           how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]
        
        # missing values (None, nan) to 0 
        user_df = user_df.fillna(-1)
        
        user_df = user_df.loc[:, ~user_df.columns.duplicated()]
        
        '''numeric vector'''
        user_x_numeric = user_df[var_numeric].to_numpy()
        
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
        
        yield user_x, user_x_numeric, label_y
#%%
'''
- user df 내에서도 sentence의 중복이 발생 (일부의 exercise data만 다르기 때문에 문제 발생)
'''
batch_size = 10
user_df_batch_gen = user_df_batch_generator(batch_size)
user_x, user_x_numeric, label_y = next(user_df_batch_gen)
user_x.shape
user_x_numeric.shape
label_y.shape
#%% input
target_num = [len(x) for x in semantic_template_dict.values()]
sas_num = len(data_num)
latent_dim = 10
#%% modeling
input_layer1 = layers.Input((sas_num, max_var_num))
w1 = layers.Dense(1, use_bias=False, activation='tanh')
attention = tf.nn.softmax((w1(input_layer1)), axis=1)
h = tf.squeeze(tf.matmul(attention, input_layer1, transpose_a=True), axis=1)

mean_dense = layers.Dense(latent_dim) 
logvar_dense = layers.Dense(latent_dim, activation='tanh') 
z_mean = mean_dense(h)
z_logvar = logvar_dense(h)
epsilon = tf.random.normal((latent_dim, ))
z = z_mean + tf.math.exp(z_logvar / 2) * epsilon 

'''decoder'''
# reconstruction y
input_layer2 = layers.Input((len(var_numeric)))
y_recon_layer = [layers.Dense(n) for n in target_num]
h = layers.concatenate((input_layer2, z)) 
y_recon = [d(h) for d in y_recon_layer]

# annotation_model = K.models.Model(input_layer, attention)
vrae = K.models.Model([input_layer1, input_layer2], [z_mean, z_logvar, z, y_recon, attention])
vrae.summary()  
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.0001):
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
        recon_loss += tf.reduce_mean(tf.multiply(scce(y[:, i], y_pred[i]), tf.cast(y[:, i] != 0, tf.float32)), keepdims=True)
    
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(logvar_pred)), axis=1), keepdims=True)
        
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%%
epochs = len(user_list)*2

'''python generator -> tf data pipeline (padding is needed)'''
user_df_batch_gen = user_df_batch_generator(batch_size) 
# dataset = tf.data.Dataset.from_generator(user_df_batch_generator, tf.float32)
# iterator = dataset.make_one_shot_iterator()
# x = iterator.get_next()

optimizer = tf.keras.optimizers.Adam(0.001)
for epoch in range(1, epochs):
    
    beta = kl_anneal(epoch, int(epochs/2))
    if epoch % 10 == 1:
        t1 = time.time()

    x_data1, x_data2, label_y = next(user_df_batch_gen)
    x_data1 = tf.convert_to_tensor(x_data1.astype(np.float32), dtype=tf.float32)
    x_data2 = tf.convert_to_tensor(x_data2.astype(np.float32), dtype=tf.float32)
    label_y = tf.convert_to_tensor(label_y, dtype=tf.int32)
    
    with tf.GradientTape(persistent=True) as tape:
        mean_pred, logvar_pred, z_pred, y_pred, attention = vrae([x_data1, x_data2])
        recon_loss, kl_loss, loss = loss_fun(label_y, y_pred, mean_pred, logvar_pred, beta)
        
    grad = tape.gradient(loss, vrae.weights)
    optimizer.apply_gradients(zip(grad, vrae.weights))
    
    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3}) VRAE loss: {:.6}'.format(epoch, t2-t1, loss.numpy()[0]))
        print('Y RECON loss: {:.6}, KL loss: {:.6}'.format(recon_loss.numpy()[0], kl_loss.numpy()[0]))
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
        user_df = pd.merge(health.loc[np.isin(health['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                           user_df, 
                           how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]
        
        # missing values to 0 (None, nan)
        user_df = user_df.fillna(-1)
        
        user_df = user_df.loc[:, ~user_df.columns.duplicated()]
        
        '''numeric vector'''
        user_x_numeric = user_df[var_numeric].to_numpy()
        
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
        user_y = user_df[['USER_ID_x', 'CNSL_SN_x', 'sentence']]
        # user_y = user_df['sentence']
        temp = [semantic_dict.get(x) for x in user_y['sentence'].to_numpy()]
        label_y = np.zeros((user_y.shape[0], len(semantic_id)))
        for i in range(len(temp)):
            label_y[i, temp[i]] = semantic_template_dict.get(temp[i]).get(user_y['sentence'].to_numpy()[i])
                
        yield user_x, user_x_numeric, user_y, label_y
#%% validation error and show result
'''attention result는 나중에!'''
user_df_gen_val = user_df_generator_val()
count_argmax = 0
count_topk = 0
n = 0
topk = 5
val_result = {'id':['id'], 'true':['true'], 0:['경고'], 1:['권장'], 2:['지식'], 3:['현상']}
for i in tqdm(range(len(user_val_list))):
    x_data1, x_data2, user_y, label_y = next(user_df_gen_val)
    x_data1 = tf.convert_to_tensor(x_data1.astype(np.float32), dtype=tf.float32)
    x_data2 = tf.convert_to_tensor(x_data2.astype(np.float32), dtype=tf.float32)
    label_y = tf.convert_to_tensor(label_y, dtype=tf.int32)
    
    _, _, _, y_inf, _ = vrae([x_data1, x_data2])
    
    y_argmax = []
    for i in range(len(semantic_id)):
        y_argmax.append(list(np.argmax(y_inf[i], axis=1)))
        
    for k in range(len(user_y)):
        answer = user_y['sentence'].iloc[k]
        answer = '({}):{}'.format(semantic_numid.get(semantic_dict.get(answer)), answer.replace(u'\xa0', ''))
        val_result['true'] = val_result.get('true') + [answer]
        
        user_id = '_'.join([str(x) for x in user_y[['USER_ID_x', 'CNSL_SN_x']].iloc[0].to_list()])
        val_result['id'] = val_result.get('id') + [user_id]
        
        for i in range(len(semantic_id)):
            pred_ = '({}):{}'.format(semantic_numid.get(semantic_dict.get(semantic_template_numdict[i].get(y_argmax[i][k]))), 
                                     semantic_template_numdict[i].get(y_argmax[i][k])).replace(u'\xa0', '')
            
            val_result[i] = val_result.get(i) + [pred_]
    
    c_a = 0 # argmax
    c_t = 0 # topk
    for k in range(len(label_y)):
        idx_ = np.where(label_y.numpy()[k, :] != 0)[0][0]    
        c_a += label_y.numpy()[k, idx_] == np.argmax(y_inf[idx_].numpy()[k, :])
        c_t += np.isin(label_y.numpy()[k, idx_], np.argsort(y_inf[idx_].numpy()[k, :])[-topk:][::-1])
        
    count_argmax += c_a
    count_topk += c_t
    
    n += len(label_y)
    
print(count_argmax / n)
print(count_topk / n)
# 0.17698695136417555
# 0.34306049822064055

with open('./result/validation_guidance_our_health_200822.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    for k in range(len(val_result[0])): 
        wr.writerow([val_result['id'][k], val_result['true'][k], val_result[0][k], val_result[1][k], val_result[2][k], val_result[3][k]])
#%%
'''random behavior'''
# new model
input_layer1 = layers.Input((sas_num, max_var_num))
input_layer2 = layers.Input((len(var_numeric)))
z_input = layers.Input(latent_dim)
h = layers.concatenate((input_layer2, z_input)) 
y_recon = [d(h) for d in y_recon_layer]
random_model = K.models.Model([input_layer1, input_layer2, z_input], [y_recon])
random_model.summary() 
#%%
def user_df_generator_specific(idx):
    # exercise data
    user_key = user_val_list[idx]
    for n, num in enumerate(data_num):
        if n == 0:
            user_df = sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True)
        else:
            user_df = pd.merge(user_df, 
                                sas_dict[n].loc[np.isin(sas_dict[n]['USER_ID_N'].to_list(), user_key)].reset_index(drop=True), 
                                how='outer', on='CNSL_SN_N')
    
    # sentence data
    user_df = pd.merge(health.loc[np.isin(health['USER_ID'].to_list(), user_key)].reset_index(drop=True), 
                        user_df, 
                        how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
    user_df = user_df[var_filter]
    
    # missing values to 0 (None, nan)
    user_df = user_df.fillna(-1)
    
    '''numeric vector'''
    user_x_numeric = user_df[var_numeric].to_numpy()
    
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
    user_y = user_df[['USER_ID_x', 'CNSL_SN_x', 'sentence']]
    # user_y = user_df['sentence']
    temp = [semantic_dict.get(x) for x in user_y['sentence'].to_numpy()]
    label_y = np.zeros((user_y.shape[0], len(semantic_id)))
    for i in range(len(temp)):
        label_y[i, temp[i]] = semantic_template_dict.get(temp[i]).get(user_y['sentence'].to_numpy()[i])
            
    return user_x, user_x_numeric, user_y, label_y
#%%
topk = 5
random_behavior = {}

k = 0
x_data1, x_data2, user_y, label_y = user_df_generator_specific(k)
answer = user_y['sentence'].iloc[0].replace(u'\xa0', '')
answer_num = semantic_template_dict.get(semantic_dict.get(answer)).get(answer)
idx = int(np.where(label_y[0] != 0)[0])

x_data1 = tf.convert_to_tensor(x_data1[[0], ...].astype(np.float32), dtype=tf.float32)
x_data2 = tf.convert_to_tensor(x_data2[[0], ...].astype(np.float32), dtype=tf.float32)
mean_inf, logvar_inf, z_inf, y_inf, attention = vrae([x_data1, x_data2])
epsilon_ = [tf.random.normal((len(x_data1), latent_dim)) for _ in range(5)]
z_ = [mean_inf + tf.math.exp(logvar_inf / 2) * e for e in epsilon_]

for r in range(len(z_)):
    temp = random_model([x_data1, x_data2, z_[r]])[0][idx].numpy()[0]
    topk_temp = np.argsort(temp)[-topk:][::-1]
    topk_prob = np.sort(temp)[-topk:][::-1]
    random_behavior[r] = random_behavior.get(r, []) + [(x,y) for x,y in zip(topk_temp, topk_prob)]
print(answer_num)
random_behavior

# 284
# {0: [(284, 4.2070456),
#   (487, 4.0376515),
#   (440, 3.7809575),
#   (15, 3.2449093),
#   (253, 2.1932926)],
#  1: [(284, 3.8431215),
#   (487, 3.8327923),
#   (15, 3.4077644),
#   (440, 3.2897065),
#   (253, 2.2004917)],
#  2: [(487, 3.802437),
#   (284, 3.6964962),
#   (15, 3.565903),
#   (440, 3.1962442),
#   (178, 2.0526593)],
#  3: [(487, 4.545193),
#   (284, 4.3362613),
#   (15, 4.2717843),
#   (440, 3.965149),
#   (311, 2.3585806)],
#  4: [(284, 3.185714),
#   (487, 3.1116033),
#   (440, 2.5225956),
#   (316, 2.3610473),
#   (15, 2.2929287)]}
#%%






