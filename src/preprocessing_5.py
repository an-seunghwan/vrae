#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:02:09 2020

@author: anseunghwan
"""


#%%
'''
- 판단 -> template 생성
- 사람별로 데이터 구분
'''
#%%
'''
- 결측치를 고려한 딥러닝 모형 (NLP에서 UNK과 같은 역할)
- 일단 기본적으로 00 데이터를 사용
- 00에도 결측치 존재
'''
#%%
import os 
os.chdir('/Users/anseunghwan/Documents/uos/generating_text')
print('current directory:', os.getcwd())
from subprocess import check_output
print('=====Data list=====')
print(check_output(["ls", "./data"]).decode("utf8"))
#%%
from sas7bdat import SAS7BDAT
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from konlpy.tag import Okt
okt = Okt()
from collections import Counter
import re
#%%
'''
- barycenter(template) data
'''
body = pd.read_csv('./data/body_template.csv')
body.head()
body.columns
# 중복된 행 제거
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
#%% template
# template = body.message
template = body.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)
#%%
'''
- key : template dictionary
'''
# body[['USER_ID', 'CNSL_SN']]
user_list = sorted(list(set(body['USER_ID'].to_list())))
len(user_list)
#%% 
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
#%% numeric and judgement info
'''
<원본 sas 데이터>
- first row is column names
'''
print('sas data: ', 0)
with SAS7BDAT('./body_final/exercise_00.sas7bdat', encoding='euc-kr') as f:
    sas = f.to_data_frame()

sas_dict = {}
sas_dict[0] = sas

'''
- USER_ID_N, CNSL_SN_N이 있는 데이터: 00, 02, 03, 04, 05, 06, 07, 09, 12
- 07, 09번 데이터: 데이터가 너무 많음
- 08, 10, 12번 데이터: 공통 key를 찾기에는 데이터가 너무 적다
'''
for n, num in enumerate(data_num):
    if n == 0: continue
    else:
        print('sas data: ', n)
        with SAS7BDAT('./body_final/exercise_01_12/exercise_{}.sas7bdat'.format(num), encoding='euc-kr') as f:
            sas = f.to_data_frame()
        sas_dict[n] = sas
#%%
'''
- variable EDA
- 문자 속성의 데이터 변수들을 확인
- missing data 처리 방식: 숫자 = -1 , 문자 = case by case (like dummy coding)
'''
# sas 00
Counter(sas_dict[0]['PHSC_LVL'].to_list()) # dummy
Counter(sas_dict[0]['MUSCLE_EXCS_PRSCRPT'].to_list()) # 제거

# sas 02
Counter(sas_dict[1]['BLOOD_PRESS'].to_list()) # '131/85'의 형식

# sas 03
Counter(sas_dict[2]['TARGET_HEART'].to_list()) # '90 ~ 130'의 형식
Counter(sas_dict[2]['DAY_RECOM_CAL'].to_list()) # ,(comma) 제거
Counter(sas_dict[2]['TARGET_EX_DAY'].to_list()) # 전부 똑같은 데이터 -> 제거
Counter(sas_dict[2]['TARGET_EX_TIME'].to_list()) # as.int
Counter(sas_dict[2]['TARGET_WALK'].to_list()) # ,(comma) 제거

# sas 04
Counter(sas_dict[3]['DAY_ENG_NEED_AM'].to_list()) # ,(comma) 제거

# sas 09
Counter(sas_dict[6]['WALK_CNT_09'].to_list()) # ,(comma) 제거 (숫자 앞에 왜  - 가 붙어있지...?)
Counter(sas_dict[6]['MOD_EXE_TM_09'].to_list()) # as.float, 소수점 처리
Counter(sas_dict[6]['CONSUME_CAL_09'].to_list()) # as.float, 소수점 처리
Counter(sas_dict[6]['MOVE_DIST_09'].to_list()) # as.float, 소수점 처리
Counter(sas_dict[6]['AVE_WALK_CNT_09'].to_list()) # ,(comma) 제거 (숫자 앞에 왜  - 가 붙어있지...?)
Counter(sas_dict[6]['AVE_MOD_EXE_TM_09'].to_list()) # as.int
Counter(sas_dict[6]['AVE_CONSUME_CAL_09'].to_list()) # as.int (숫자 앞에 왜  - 가 붙어있지...?)
Counter(sas_dict[6]['AVE_MOVE_DIST_09'].to_list()) # as.int
Counter(sas_dict[6]['ACT_DAYS_09'].to_list()) # as.float, 소수점 처리

# sas 12
Counter(sas_dict[7]['TOT_EXCS_TM'].to_list()) # as.float, 소수점 처리
Counter(sas_dict[7]['EXCS_TM'].to_list()) # as.float, 소수점 처리
Counter(sas_dict[7]['EXCS_RATE'].to_list()) # as.float, 소수점 처리
#%%
'''
- character variable preprocessing
'''
# sas 00
phsc_lvl_dict = {'': 0, 'P200' : 1, 'P250' : 2, 'P300' : 3, 'P350' : 4, 'P400' : 5}
phsc_lvl = sas_dict[0]['PHSC_LVL'].map(lambda x : phsc_lvl_dict.get(x)).to_numpy() # dummy
sas_dict[0]['PHSC_LVL'] = sas_dict[0]['PHSC_LVL'].map(lambda x : phsc_lvl_dict.get(x)) # dummy
# Counter(sas_dict[0]['MUSCLE_EXCS_PRSCRPT'].to_list()) # 변수 제거

# sas 02
Counter(sas_dict[1]['BLOOD_PRESS'].to_list()) # '131/85'의 형식 -> split lower and upper & 변수 제거
sas_dict[1].insert(list(sas_dict[1].columns).index('BLOOD_PRESS')+1, "BLOOD_PRESS_L", 
                   sas_dict[1]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[1]))
sas_dict[1].insert(list(sas_dict[1].columns).index('BLOOD_PRESS')+2, "BLOOD_PRESS_U", 
                   sas_dict[1]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[0]))
sas_dict[1]['BLOOD_PRESS_L'] = sas_dict[1]['BLOOD_PRESS_L'].map(lambda x : float(x)) 
sas_dict[1]['BLOOD_PRESS_U'] = sas_dict[1]['BLOOD_PRESS_U'].map(lambda x : float(x)) 

# sas 03
Counter(sas_dict[2]['TARGET_HEART'].to_list()) # '90 ~ 130'의 형식 -> split lower and upper & 변수 제거
sas_dict[2].insert(list(sas_dict[2].columns).index('TARGET_HEART')+1, "TARGET_HEART_L", 
                   sas_dict[2]['TARGET_HEART'].map(lambda x : x.split(' ~ ')).map(lambda x : x[0]))
sas_dict[2].insert(list(sas_dict[2].columns).index('TARGET_HEART')+2, "TARGET_HEART_U", 
                   sas_dict[2]['TARGET_HEART'].map(lambda x : x.split(' ~ ')).map(lambda x : x[1]))
sas_dict[2]['TARGET_HEART_L'] = sas_dict[2]['TARGET_HEART_L'].map(lambda x : float(x)) 
sas_dict[2]['TARGET_HEART_U'] = sas_dict[2]['TARGET_HEART_U'].map(lambda x : float(x)) 

sas_dict[2]['DAY_RECOM_CAL'] = sas_dict[2]['DAY_RECOM_CAL'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거
# Counter(sas_dict[2]['TARGET_EX_DAY'].to_list()) # 전부 똑같은 데이터 -> 변수 제거
sas_dict[2]['TARGET_EX_TIME'] = sas_dict[2]['TARGET_EX_TIME'].map(lambda x : float(x)) # as.int
sas_dict[2]['TARGET_WALK'] = sas_dict[2]['TARGET_WALK'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거

# sas 04
sas_dict[3]['DAY_ENG_NEED_AM'] = sas_dict[3]['DAY_ENG_NEED_AM'].map(lambda x : int(re.sub(',', '', x))) # ,(comma) 제거 

# sas 09
sas_dict[6]['WALK_CNT_09'] = sas_dict[6]['WALK_CNT_09'].map(lambda x : str(x).replace(',', '').replace('-', '') 
                                                            if x != '###########' else str(0)) # ,(comma) 제거 (숫자 앞에 왜  - 가 붙어있지...?)
sas_dict[6]['WALK_CNT_09'] = sas_dict[6]['WALK_CNT_09'].map(lambda x : 0 if len(x) == 0 else float(x)) 
sas_dict[6]['MOD_EXE_TM_09'] = sas_dict[6]['MOD_EXE_TM_09'].map(lambda x : float(x) if x != '##########' else 0) # as.float, 소수점 처리
sas_dict[6]['CONSUME_CAL_09'] = sas_dict[6]['CONSUME_CAL_09'].map(lambda x : float(x) if x != '##########' else 0) # as.float, 소수점 처리
sas_dict[6]['MOVE_DIST_09'] = sas_dict[6]['MOVE_DIST_09'].map(lambda x : float(x) if x != '##########' else 0) # as.float, 소수점 처리
sas_dict[6]['AVE_WALK_CNT_09'] = sas_dict[6]['AVE_WALK_CNT_09'].map(lambda x : str(x).replace(',', '').replace('-', '') 
                                                            if x != '###########' else str(0)) # ,(comma) 제거 (숫자 앞에 왜  - 가 붙어있지...?)
sas_dict[6]['AVE_WALK_CNT_09'] = sas_dict[6]['AVE_WALK_CNT_09'].map(lambda x : 0 if len(x) == 0 else float(x)) 
sas_dict[6]['AVE_MOD_EXE_TM_09'] = sas_dict[6]['AVE_MOD_EXE_TM_09'].map(lambda x : int(x)) # as.int
sas_dict[6]['AVE_CONSUME_CAL_09'] = sas_dict[6]['AVE_CONSUME_CAL_09'].map(lambda x : str(x).replace('-', '') 
                                                            if x != '##########' else str(0)) # as.int (숫자 앞에 왜  - 가 붙어있지...?)
sas_dict[6]['AVE_CONSUME_CAL_09'] = sas_dict[6]['AVE_CONSUME_CAL_09'].map(lambda x : 0 if len(x) == 0 else float(x)) 
sas_dict[6]['AVE_MOVE_DIST_09'] = sas_dict[6]['AVE_MOVE_DIST_09'].map(lambda x : float(x)) # as.int
sas_dict[6]['ACT_DAYS_09'] = sas_dict[6]['ACT_DAYS_09'].map(lambda x : float(x)) # as.float, 소수점 처리

# sas 12
sas_dict[7]['TOT_EXCS_TM'] = sas_dict[7]['TOT_EXCS_TM'].map(lambda x : float(x)) # as.float, 소수점 처리
sas_dict[7]['EXCS_TM'] = sas_dict[7]['EXCS_TM'].map(lambda x : float(x)) # as.float, 소수점 처리
sas_dict[7]['EXCS_RATE'] = sas_dict[7]['EXCS_RATE'].map(lambda x : float(x)) # as.float, 소수점 처리
#%%
'''data summary'''
for i in range(len(sas_dict)):
    temp = sas_dict.get(i).describe()
    temp.to_csv('./data/sas_{}_summary.csv'.format(data_num[i]), encoding='euc_kr')
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
var_filter_kor = [(x, y) for x,y in datamart_dict.items() if x in var_filter]
var_filter_kor
#%%
'''data scaling
-> user_df 모두 통합 후 scaling 진행
'''
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

sas_dict_scaled = deepcopy(sas_dict)
# missing values to 0 (None, nan) & scaling
for i, num in enumerate(data_num):
    scaler = StandardScaler()
    if i == 0:
        scaled_sas = scaler.fit_transform(sas_dict[i][globals()['var{}'.format(num)]].fillna(-1))
        scaled_sas[:, 0] = phsc_lvl
        sas_dict_scaled[i][globals()['var{}'.format(num)]] = scaled_sas
    else:
        scaled_sas = scaler.fit_transform(sas_dict[i][globals()['var{}'.format(num)]].fillna(-1))
        sas_dict_scaled[i][globals()['var{}'.format(num)]] = scaled_sas

#%%
def user_df_generator():
    i = 0
    while i < len(user_list):
        # exercise data
        user_key = user_list[i]
        for n, num in enumerate(data_num):
            if n == 0:
                user_df = sas_dict_scaled[n].loc[sas_dict_scaled[n]['USER_ID_N'] == user_key].reset_index(drop=True)
            else:
                user_df = pd.merge(user_df, sas_dict_scaled[n].loc[sas_dict_scaled[n]['USER_ID_N'] == user_key].reset_index(drop=True), how='outer', on='CNSL_SN_N')
        # sentence data
        user_df = pd.merge(body.loc[body['USER_ID'] == user_key].reset_index(drop=True), user_df, how='inner', left_on='CNSL_SN', right_on='CNSL_SN_N')
        user_df = user_df[var_filter]

        # missing values to 0 (None, nan)
        # user_df = user_df.fillna(-1)
        i += 1
        yield user_df
#%%
'''
- user df 내에서도 sentence의 중복이 발생 (일부의 exercise data만 다르기 때문에 문제 발생)
'''
user_df_gen = user_df_generator()
user_df = next(user_df_gen)
user_df.to_csv('./data/user_df_sample_scaled.csv', encoding='euc_kr')
#%%
'''save sas dict'''
import pickle
with open("./body_final/sas_dict_scaled.pkl", "wb") as f:
    pickle.dump(sas_dict_scaled, f)

# with open("./body_final/sas_dict.pkl", "rb") as f:
    # dict_ = pickle.load(f)
#%%
'''
(barycenter template, exercise) pair
-> 비효율적 수정필요
'''
# body template dictionary
id_sn = [0]*len(body)
for i in tqdm(range(len(body))):
    id_sn[i] = '_'.join([str(int(x)) for x in body[['USER_ID', 'CNSL_SN']].iloc[i].to_list()])
idsn_template_dict = {x:y for x,y in zip(id_sn, body.sentence)}
len(idsn_template_dict)
#%%
# body info dictionary (sas data)
idsn_sas_dict = {}
for n, num in enumerate(data_num):
    sas = sas_dict.get(n)
    id_sn = [0]*len(sas)
    # if 'CNSL_SN_N' in sas.columns:
    for i in tqdm(range(len(sas))):
        id_sn[i] = '_'.join([str(int(x)) for x in sas[['USER_ID_N', 'CNSL_SN_N']].iloc[i].to_list()])
    # dict_ = {id_sn[i]:sas.iloc[i][4:].to_list() for i in range(len(sas))}
    dict_ = {id_sn[i]:sas.iloc[i].to_list() for i in range(len(sas))}
    idsn_sas_dict[n] = dict_
#%%
for i, keys in enumerate(idsn_sas_dict.values()):
    if i == 0:
        matched_idsn = set(keys)
    else:    
        matched_idsn = matched_idsn & set(keys)
matched_idsn = matched_idsn & set(idsn_template_dict.keys())
matched_idsn = list(matched_idsn)
len(matched_idsn)
#%%
j = 10000
pair = []
for k, v in idsn_sas_dict.items():
    pair += ['('+x+'): '+str(y) for x,y in zip(datamart_dict.get(k).values(), v.get(matched_idsn[j]))]
pair = ['template: ' + idsn_template_dict.get(matched_idsn[j])] + pair
pair
#%%





























