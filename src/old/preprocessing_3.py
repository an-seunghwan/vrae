#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:02:09 2020

@author: anseunghwan
"""


#%%
'''
- 판단 -> template 생성
'''
#%%
'''
- 결측치를 고려한 딥러닝 모형
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
#%%
'''
- barycenter data
'''
body = pd.read_csv('./data/body_template.csv')
body.head()
body.columns
#%% template
# template = body.message
template = body.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)
#%% numeric and judgement info
'''
<원본 sas 데이터>
- first row is column names
'''
data = []
with SAS7BDAT('./body_final/exercise_00.sas7bdat', encoding='euc-kr') as f:
    for i, row in enumerate(f):
        if len(row) == 0: break
        data.append(row)
        # if i == 20000: break
        
# print(data[0])
# print(data[1])

sas = pd.DataFrame(columns=data[0])
sas = sas.append(pd.DataFrame(data[1:], columns=data[0]))
# df.head()
# df.iloc[10]

sas_dict = {}
sas_dict[0] = sas
#%%
'''
- USER_ID_N, CNSL_SN_N이 있는 데이터: 00, 02, 03, 04, 05, 06, 07, 09, 12
- 07, 09번 데이터: 데이터가 너무 많음
- 08, 10, 12번 데이터: 공통 key를 찾기에는 데이터가 너무 적다
'''
# 07번, 09번 sas 데이터가 엄청 큼!!!
# data_num = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '12']
data_num = ['00', '02', '03', '04', '05', '06']
for n, num in enumerate(data_num):
    if n == 0: continue
    else:
        print('sas data: ', n)
        data = []
        with SAS7BDAT('./body_final/exercise_01_12/exercise_{}.sas7bdat'.format(num), encoding='euc-kr') as f:
            for i, row in enumerate(f):
                if len(row) == 0: break
                data.append(row)
                # if i == 20000: break
    
        sas = pd.DataFrame(columns=data[0])
        sas = sas.append(pd.DataFrame(data[1:], columns=data[0]))
        # df02.head()
        # df02.iloc[10]
        sas_dict[n] = sas
#%% 
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
datamart_dict = {}
for n, num in enumerate(data_num):
    # if n == 0:
    #     datamart = pd.read_csv('./body_final/body_datamart_V1_{}.csv'.format(num), skiprows=2).iloc[:-2]
    # else:
    #     datamart = pd.read_csv('./body_final/body_datamart_V1_{}.csv'.format(num), skiprows=5).iloc[:-2]
    datamart = pd.read_csv('./body_final/body_datamart_V1_{}.csv'.format(num), skiprows=0).iloc[:-2]
    dict_ = {x:y for x,y in zip(datamart['변수 이름'], datamart['레이블'])}
    datamart_dict[n] = dict_
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





























