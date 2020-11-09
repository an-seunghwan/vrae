#%%
'''
- Time Trending Attention
- 한 user에 대해서 numeric 데이터를 time sequence로 입력
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
import pickle
from copy import deepcopy
import os
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%%
'''
barycenter(template) data loading
'''
body = pd.read_csv('./data/barycenter_body.csv')
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
idx = np.random.choice(range(len(user_list_total)), int(len(user_list_total) * 0.9), replace=False)
user_list = list(np.array(user_list_total)[idx])
user_val_list = [x for x in user_list_total if x not in user_list]

'''sas dict data'''
with open("./body_final/sas_dict_scaled.pkl", "rb") as f:
    sas_dict = pickle.load(f)
#%%    
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
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
var_numeric = var00 + var02 + var03 + var04 + var05 + var06 + var09 + var12

# what is filtered?
# var_filter_kor = [(x, y) for x,y in datamart_dict.items() if x in var_filter]
# var_filter_kor

data_num = ['00', '02', '03', '04', '05', '06', '09', '12']
max_var_num = max(len(globals()['var{}'.format(num)]) for num in data_num)
#%%
def user_df_generator(user_list_):
    for i in range(len(user_list_)):
        # idx = np.random.randint(0, len(user_list)) # permutation
        user_key = user_list_[i]
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

        # sort by CNSL_SN
        user_df = user_df.sort_values(by=['CNSL_SN'], axis=0)

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

        yield user_x, user_x_numeric, label_y, user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        # yield user_x, label_y
#%%
userdfgen = user_df_generator(user_list)
user_x, user_x_numeric, label_y, user_y = next(userdfgen)
user_x.shape
user_x_numeric.shape
label_y.shape
user_y.shape
#%%
# userdfgen = user_df_generator(user_list)
# seqlen = []
# for i in tqdm(range(len(user_list))):
#     _, _, label_y, _ = next(userdfgen)
#     seqlen.append(len(label_y))
#%%
# import matplotlib.pyplot as plt
# plt.hist(seqlen, bins=20)
# np.quantile(np.array(seqlen), 0.75) # 56
maxlen = 56
#%% 
userdfgen = user_df_generator(user_list)
user_x, user_x_numeric, label_y, _ = next(userdfgen)
# user_x, label_y = next(userdfgen)
user_x.shape
user_x_numeric.shape
label_y.shape

user_x = preprocessing.sequence.pad_sequences(user_x[np.newaxis, ...], maxlen=maxlen, padding='post')
user_x.shape
user_x_ = [user_x[..., i, :] for i in range(user_x.shape[2])]
user_x_[0].shape
user_x_numeric = preprocessing.sequence.pad_sequences(user_x_numeric[np.newaxis, ...], maxlen=maxlen, padding='post')
user_x_numeric.shape
label_y = preprocessing.sequence.pad_sequences(label_y[np.newaxis, ...], maxlen=maxlen, padding='post')
label_y.shape

# index_y = deepcopy(label_y)
# index_y[np.where(label_y != 0)] = 1
#%%
target_num = [len(x) for x in semantic_template_dict.values()]
sas_num = len(data_num)
latent_dim = 10
units = 10
#%% 
'''modeling
- time trending attention (RNN attention)
'''
# class AttentionRNNCell(K.layers.Layer):

#     def __init__(self, units, data_num, **kwargs):
#         self.units = units
#         self.state_size = units
#         self.data_num = data_num
#         super(AttentionRNNCell, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(shape=(input_shape[-1], 1),
#                                       initializer='glorot_uniform',
#                                       name='kernel')
#         self.recurrent_kernel = self.add_weight(
#             shape=(self.data_num, self.data_num),
#             initializer=K.initializers.Identity(gain=1.0),
#             name='recurrent_kernel')
#         self.built = True

#     def call(self, inputs, states):
#         prev_output = states[0]
#         h = tf.matmul(inputs, self.kernel)
#         output = tf.tanh(h - tf.matmul(prev_output, self.recurrent_kernel))
#         return output, [output]
#%%
# attentionrnncell = AttentionRNNCell(units, sas_num)
# layer = layers.RNN(attentionrnncell)
# layer(user_x)
#%%
user_input = layers.Input((maxlen, sas_num, max_var_num)) # matrix
user_input1 = [layers.Input((maxlen, max_var_num)) for _ in range(sas_num)] # matrix
user_input2 = layers.Input((maxlen, len(var_numeric))) # vector

RNN_layer = layers.SimpleRNN(units=1, activation='tanh', use_bias=False, return_sequences=True) 
hidden = [RNN_layer(x) for x in user_input1]
h = hidden[0]
for i in range(1, len(hidden)):
    h = layers.Concatenate(axis=-1)([h, hidden[i]])
attention = tf.nn.softmax(h, axis=-1)[:, :, tf.newaxis, :]
h = tf.squeeze(tf.matmul(attention, user_input), axis=2)

mean_dense = layers.Dense(latent_dim) 
logvar_dense = layers.Dense(latent_dim, activation='tanh') 
z_mean = mean_dense(h)
z_logvar = logvar_dense(h)
epsilon = tf.random.normal((maxlen, latent_dim))
z = z_mean + tf.math.exp(z_logvar / 2) * epsilon 

y_recon_layer = [layers.Dense(n) for n in target_num]
h = layers.concatenate((user_input2, z)) 
y_recon = [d(h) for d in y_recon_layer]

trend_model = K.models.Model([user_input, user_input1, user_input2], [z_mean, z_logvar, z, y_recon, attention])
trend_model.summary()  
#%%
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction=tf.keras.losses.Reduction.NONE)
def loss_fun(y, y_pred, mean, logvar, beta):
    '''do not consider NONE'''
    # reconstruction loss
    recon_loss = tf.zeros(())
    for i in range(len(semantic_id)):
        recon_loss += tf.divide(tf.reduce_sum(tf.multiply(scce(y[0, :, i], y_pred[i]), tf.cast(y[0, :, i] != 0, tf.float32)), keepdims=True),
                                tf.reduce_sum(tf.cast(y[0, :, i] != 0, tf.float32)) + 0.00001)
    
    # kl-divergence loss
    kl_loss = tf.reduce_sum(tf.reduce_sum(-0.5 * (1 + logvar_pred - tf.math.pow(mean_pred, 2) - tf.math.exp(logvar_pred)), axis=-1), keepdims=True)
        
    return [recon_loss, kl_loss, recon_loss + beta * kl_loss]
#%%
'''
- kl annealing using logistic vs linear
'''
def kl_anneal(step, s, k=0.0005):
    # logistic
    return 1 / (1 + math.exp(-k*(step - s)))
#%% training 
optimizer = tf.keras.optimizers.Adam(0.001)
userdfgen = user_df_generator(user_list)
epochs = len(user_list) * 100
for epoch in range(epochs):
    if epoch % 10 == 0:
        t1 = time.time()
        
    beta = kl_anneal(epoch, int(epochs/2))
    
    x_data1, x_data2, label_y, _ = next(userdfgen)    
    x_data = preprocessing.sequence.pad_sequences(x_data1[np.newaxis, ...], maxlen=maxlen, padding='post')
    x_data1 = [x_data[..., i, :] for i in range(x_data.shape[2])]
    x_data2 = preprocessing.sequence.pad_sequences(x_data2[np.newaxis, ...], maxlen=maxlen, padding='post')
    label_y = preprocessing.sequence.pad_sequences(label_y[np.newaxis, ...], maxlen=maxlen, padding='post')
        
    with tf.GradientTape(persistent=True) as tape:
        mean_pred, logvar_pred, z_pred, y_pred, _ = trend_model([x_data, x_data1, x_data2])
        recon_loss, kl_loss, loss = loss_fun(label_y, y_pred, mean_pred, logvar_pred, beta)
            
    grad = tape.gradient(loss, trend_model.weights)
    optimizer.apply_gradients(zip(grad, trend_model.weights))
    
    if epoch % 10 == 1:
        t2 = time.time()
        print('({} epoch, time: {:.3}) loss: {:.6}'.format(epoch, t2-t1, loss.numpy()[0, 0]))
        print('Y RECON loss: {:.6}, KL loss: {:.6}'.format(recon_loss.numpy()[0, 0], kl_loss.numpy()[0, 0]))
    
    if (epoch + 1) % len(user_list) == 0:
        userdfgen = user_df_generator(user_list)
#%% validation error and show result
'''attention result는 나중에!'''
userdfgen_val = user_df_generator(user_val_list)
count_argmax = 0
count_topk = 0
n = 0
topk = 5
val_result = {'id':['id'], 'true':['true'], 0:['경고'], 1:['권장'], 2:['지식'], 3:['현상']}
for i in tqdm(range(len(user_val_list))):
    x_data1, x_data2, label_y, user_y = next(userdfgen_val)    
    x_data = preprocessing.sequence.pad_sequences(x_data1[np.newaxis, ...], maxlen=maxlen, padding='post')
    x_data1 = [x_data[..., i, :] for i in range(x_data.shape[2])]
    x_data2 = preprocessing.sequence.pad_sequences(x_data2[np.newaxis, ...], maxlen=maxlen, padding='post')
    label_y = preprocessing.sequence.pad_sequences(label_y[np.newaxis, ...], maxlen=maxlen, padding='post')
    
    _, _, _, y_inf, _ = trend_model([x_data, x_data1, x_data2])
    
    nonpad = max([i for i,x in enumerate(np.sum(label_y[0, ...], axis=1)) if x != 0]) + 1
    
    y_argmax = []
    for i in range(len(semantic_id)):
        y_argmax.append(list(np.argmax(y_inf[i][0, :nonpad, :], axis=-1)))
        
    for k in range(min(len(user_y), nonpad)):
        answer = user_y['sentence'].iloc[k].replace(u'\xa0', '')
        answer = '({}):{}'.format(semantic_numid.get(semantic_dict.get(answer)), answer)
        val_result['true'] = val_result.get('true') + [answer]
        
        user_id = '_'.join([str(x) for x in user_y[['USER_ID', 'CNSL_SN']].iloc[k].to_list()])
        val_result['id'] = val_result.get('id') + [user_id]
        
        for i in range(len(semantic_id)):
            pred_ = '({}):{}'.format(semantic_numid.get(semantic_dict.get(semantic_template_numdict[i].get(y_argmax[i][k]))), 
                                     semantic_template_numdict[i].get(y_argmax[i][k])).replace(u'\xa0', '')
            
            val_result[i] = val_result.get(i) + [pred_]
    
    c_a = 0 # argmax
    c_t = 0 # topk
    for k in range(nonpad):
        idx_ = np.where(label_y[0, k, :] != 0)[0][0]    
        c_a += label_y[0, k, idx_] == np.argmax(y_inf[idx_].numpy()[0, k, :])
        c_t += np.isin(label_y[0, k, idx_], np.argsort(y_inf[idx_].numpy()[0, k, :])[-topk:][::-1])
        
    count_argmax += c_a
    count_topk += c_t
    
    n += nonpad
    
print(count_argmax / n)
print(count_topk / n)
# 0.43575358722623797
# 0.7555291832991693
#%%
with open('./result/valguidance_trending_our_200828.csv', 'w', encoding='euc-kr', newline='') as f:
    wr = csv.writer(f)
    for k in range(len(val_result[0])): 
        wr.writerow([val_result['id'][k], val_result['true'][k], val_result[0][k], val_result[1][k], val_result[2][k], val_result[3][k]])
#%%