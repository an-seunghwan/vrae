#%%
'''
- 한 user에 대해서 numeric 데이터를 time sequence로 입력
'''
#%%
import tensorflow as tf
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
barycenter(template) data loading
'''
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
