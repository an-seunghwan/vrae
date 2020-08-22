#%%
'''
- X token <- replacing with numeric info
- template + x = message
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
import csv    
import time
import re
from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
import pickle
import os
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%%
'''data loading'''
'''barycenter(template) data'''
body = pd.read_csv('./data/barycenter_body.csv')
body.head()
body.columns
#%%
'''X token이 있는 template 데이터만을 추출'''
x_token_idx = [i for i,x in enumerate(body['sentence']) if re.search('X', x)]
len(x_token_idx)
body = body[['USER_ID', 'CNSL_SN', 'sentence', 'message']].iloc[x_token_idx]
#%%
'''user id list'''
user_list_total = sorted(list(set(body['USER_ID'].to_list())))
len(user_list_total)
np.random.seed(1)
idx = np.random.choice(range(len(user_list_total)), int(len(user_list_total) * 0.9), replace=False)
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
var_filter = ['USER_ID', 'CNSL_SN', 'sentence', 'message'] + var00 + var02 + var03 + var04 + var05 + var06 + var09 + var12

# data_num = ['00', '02', '03', '04', '05', '06', '09', '12']
max_var_num = max(len(globals()['var{}'.format(num)]) for num in data_num)
#%%
'''vocab을 우선 생성해 놓고 시작'''
unique_template = list(set(body['sentence'].to_list()))
unique_template_tokenized = [0]*len(unique_template)
for i in tqdm(range(len(unique_template))):
    unique_template_tokenized[i] = [x[0] for x in okt.pos(unique_template[i])]

template_vocab = set()
for i in tqdm(range(len(unique_template_tokenized))):
    template_vocab.update(unique_template_tokenized[i])

template_vocab = {x:i+4 for i,x in enumerate(sorted(list(template_vocab)))}
template_vocab['<PAD>'] = 0
template_vocab['<UNK>'] = 1
template_vocab['<sos>'] = 2
template_vocab['<eos>'] = 3

template_vocabsize = len(template_vocab)
template_vocabsize

template_maxlen = max(len(x) for x in unique_template_tokenized)

template_numvocab = {y:x for x,y in template_vocab.items()}
#%%
unique_message = list(set(body['message'].to_list()))
unique_message_tokenized = [0]*len(unique_message)
for i in tqdm(range(len(unique_message))):
    unique_message_tokenized[i] = [x[0] for x in okt.pos(unique_message[i])]

message_vocab = set()
for i in tqdm(range(len(unique_message_tokenized))):
    message_vocab.update(unique_message_tokenized[i])

message_vocab = {x:i+4 for i,x in enumerate(sorted(list(message_vocab)))}
message_vocab['<PAD>'] = 0
message_vocab['<UNK>'] = 1
message_vocab['<sos>'] = 2
message_vocab['<eos>'] = 3

message_vocabsize = len(message_vocab)
message_vocabsize

message_maxlen = max(len(x) for x in unique_message_tokenized)

message_numvocab = {y:x for x,y in message_vocab.items()}
#%%
def user_df_generator():
    for i in range(len(user_list)):
        # exercise data
        user_key = user_list[i]
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
        
        '''template and message sequence'''
        t_seq = [[2] + [template_vocab.get(x[0]) for x in okt.pos(sent)] + [3] for sent in user_df['sentence']]
        m_seq = [[2] + [message_vocab.get(x[0]) for x in okt.pos(sent)] + [3] for sent in user_df['message']]
        t_seq = preprocessing.sequence.pad_sequences(t_seq,
                                                    maxlen=template_maxlen,
                                                    padding='post',
                                                    value=0)
        m_seq = preprocessing.sequence.pad_sequences(m_seq,
                                                    maxlen=message_maxlen,
                                                    padding='post',
                                                    value=0)

        
        # user 데이터가 존재하지 않을 수 있음! 주의!
        
        yield user_x, t_seq, m_seq
#%%
def user_df_batch_generator():
    while True:
        idx = np.random.randint(0, len(user_list), size=batch_size)
        user_key = list(np.array(user_list)[idx])
        
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
        
        '''template and message sequence'''
        t_seq = [[2] + [template_vocab.get(x[0]) for x in okt.pos(sent)] + [3] for sent in user_df['sentence']]
        m_seq = [[2] + [message_vocab.get(x[0]) for x in okt.pos(sent)] + [3] for sent in user_df['message']]
        t_seq = preprocessing.sequence.pad_sequences(t_seq,
                                                    maxlen=template_maxlen,
                                                    padding='post',
                                                    value=0)
        m_seq = preprocessing.sequence.pad_sequences(m_seq,
                                                    maxlen=message_maxlen,
                                                    padding='post',
                                                    value=0)
        
        # user 데이터가 존재하지 않을 수 있음! 주의!
        
        yield user_x, t_seq, m_seq
#%%
# user_df_gen = user_df_generator()
# user_x, template_sequence, message_sequence = next(user_df_gen)
# user_x.shape
# template_sequence.shape
# message_sequence.shape
#%% parameters
batch_size = 10
embedding_size = 20
latent_dim = 20
units = 16
#%% encoder
t = layers.Input((template_maxlen))
# embedding_layer = layers.Embedding(input_dim=vocab_size,
#                                     output_dim=embedding_size)
t_embedding_layer = layers.Embedding(input_dim=template_vocabsize,
                                     output_dim=embedding_size,
                                       mask_zero=True)

et = t_embedding_layer(t)
encoder_lstm = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
encoder_h = encoder_lstm(et)
ew1 = layers.Dense(1, activation='tanh')
attention = tf.nn.softmax(ew1(encoder_h), axis=1)
h = tf.squeeze(tf.matmul(attention, encoder_h, transpose_a=True), axis=1)

input_layer = layers.Input((len(data_num), max_var_num))
x = tf.reshape(input_layer, (-1, len(data_num)*max_var_num))

mean_dense = layers.Dense(latent_dim)
logvar_dense = layers.Dense(latent_dim, activation='tanh') 
z_mean = mean_dense(tf.concat((h, x), axis=-1))
z_logvar = logvar_dense(tf.concat((h, x), axis=-1))
epsilon = tf.random.normal((latent_dim, ))
z = z_mean + tf.math.exp(z_logvar / 2) * epsilon
#%% decoder
m = layers.Input((message_maxlen-1))
m_embedding_layer = layers.Embedding(input_dim=message_vocabsize,
                                     output_dim=embedding_size,
                                       mask_zero=True)
em = m_embedding_layer(m)

decoder_lstm = layers.LSTM(units, 
                           return_sequences=True)
decoder_h = decoder_lstm(em)

encoder_h_ = layers.RepeatVector(message_maxlen-1)(tf.reshape(et, (-1, template_maxlen*embedding_size)))
x_ = layers.RepeatVector(message_maxlen-1)(x)
logit_layer = layers.TimeDistributed(layers.Dense(message_vocabsize)) # no softmax normalizaing -> logit tensor (from_logits=True)
logit = logit_layer(tf.concat((decoder_h, encoder_h_, x_), axis=-1))
#%% model
text_vae = K.models.Model([t, input_layer, m], [z_mean, z_logvar, z, logit])
text_vae.summary()  
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.0005):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%% loss
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.NONE)
def loss_fun(y, y_pred, mean, logvar, beta):
    '''do not consider padding'''
    # reconstruction loss
    non_pad_count = tf.reduce_sum(tf.cast(tf.cast(y != 0, tf.bool), tf.float32), axis=1, keepdims=True)
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.divide(tf.multiply(scce(y, y_pred), 
                                                                    tf.cast(tf.cast(y != 0, tf.bool), tf.float32)), 
                                                        non_pad_count), axis=1), keepdims=True)
    # kl-divergence loss
    kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5 * (1 + logvar - tf.math.pow(mean, 2) - tf.math.exp(logvar)), axis=1), keepdims=True)
        
    return recon_loss, kl_loss, recon_loss + beta * kl_loss
#%% training 
optimizer = tf.keras.optimizers.Adam(0.0005)
user_df_gen = user_df_batch_generator()
epochs = len(user_list)
for epoch in range(1, epochs+1):
    if epoch % 10 == 1:
        t1 = time.time()
        
    beta = kl_anneal(epoch, int(epochs/2))
    user_x, template_sequence, message_sequence = next(user_df_gen)    
    
    with tf.GradientTape(persistent=True) as tape:
        mean_pred, logvar_pred, z_pred, y_pred = text_vae([template_sequence, user_x, message_sequence[:, :-1]])
        recon_loss, kl_loss, loss = loss_fun(message_sequence[:, 1:], y_pred, mean_pred, logvar_pred, beta)
        
    grad = tape.gradient(loss, text_vae.weights)
    optimizer.apply_gradients(zip(grad, text_vae.weights))
    
    if epoch % 50 == 0:
        pred_id = tf.argmax(y_pred[:5], axis=-1).numpy()
        result = [0]*5
        for i in range(5):
            result[i] = ' '.join([message_numvocab.get(x) for x in pred_id[i]])
        pprint(result)

    if epoch % 10 == 0:
        t2 = time.time()
        print('({} epoch, time: {:.3}) textVAE loss: {:.6}'.format(epoch, t2-t1, loss.numpy()[0]))
        print('message loss: {:.6}, KL loss: {:.6}'.format(recon_loss.numpy()[0], kl_loss.numpy()[0]))
#%%
# K.backend.clear_session()
#%%
def user_df_val_generator():
    for i in range(len(user_val_list)):
        # exercise data
        user_key = user_list[i]
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
        
        '''template and message sequence'''
        t_seq = [[2] + [template_vocab.get(x[0]) for x in okt.pos(sent) if x[0] in template_vocab] + [3] 
                 for sent in user_df['sentence']]
        m_seq = [[2] + [message_vocab.get(x[0]) for x in okt.pos(sent) if x[0] in message_vocab] + [3] 
                 for sent in user_df['message']]
        t_seq = preprocessing.sequence.pad_sequences(t_seq,
                                                    maxlen=template_maxlen,
                                                    padding='post',
                                                    value=0)
        m_seq = preprocessing.sequence.pad_sequences(m_seq,
                                                    maxlen=message_maxlen,
                                                    padding='post',
                                                    value=0)

        
        # user 데이터가 존재하지 않을 수 있음! 주의!
        
        yield user_x, t_seq, m_seq
#%%
# user_df_val_gen = user_df_val_generator()
# user_x, template_sequence, message_sequence = next(user_df_val_gen) 
# template_sequence.shape
# message_sequence.shape
#%%
user_df_val_gen = user_df_val_generator()
true_message_list = []
result_list = []
template_list = []
for j in tqdm(range(len(user_val_list))):
    user_x, template_sequence, message_sequence = next(user_df_val_gen)
    val_seq = np.zeros((len(template_sequence), message_maxlen-1))
    val_seq[:, 0] = message_vocab.get('<sos>')
    result = ['']*len(template_sequence)

    for timestep in range(1, message_maxlen-1):
        _,_,_,pred = text_vae([template_sequence, user_x, tf.cast(val_seq, tf.float32)])
        # argmax
        pred_num = np.argsort(pred[:, timestep-1], axis=1)[:, -1]
        # topk
        # pred_ = np.argsort(pred[:, timestep-1], axis=1)[:, -3:][:, ::-1]
        # pred_num = [np.random.choice(x) for x in pred_]
        result = [result[i] + message_numvocab.get(x) + ' ' for i,x in enumerate(pred_num)]
        val_seq[:, timestep] = pred_num

    temp_ = [' '.join([template_numvocab.get(x) for x in template_sequence[i]]) for i in range(len(template_sequence))]
    temp = [temp_[i].replace('<PAD>', '').strip() for i in range(len(temp_))]
    template_list.extend(temp)
    
    # print('=====true=====')
    true_message_ = [' '.join([message_numvocab.get(x) for x in message_sequence[i]]) for i in range(len(message_sequence))]
    true_message = [true_message_[i].replace('<PAD>', '').strip() for i in range(len(true_message_))]
    
    # pprint(true_message)
    # print('\n=====pred=====')
    for i in range(len(result)):
        if re.search(' <eos>', result[i]):
            result[i] = result[i][:re.search(' <eos>', result[i]).span()[0]]
    
    assert len(true_message) == len(result)
            
    true_message_list.extend(true_message)
    result_list.extend(result)
#%% save
with open('./result/template2message_argmax_200815.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['template', 'pred', 'true'])
    for k in range(len(true_message_list)): 
        wr.writerow([template_list[k], result_list[k], true_message_list[k]])
#%%