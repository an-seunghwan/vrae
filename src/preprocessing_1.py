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
        data.append(row)
        if i == 20000: break
# print(data[0])
# print(data[1])

df00 = pd.DataFrame(columns=data[0])
df00 = df00.append(pd.DataFrame(data[1:], columns=data[0]))
df00.head()
df00.iloc[10]
# columns selection
col00 = []
#%%
data = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
with SAS7BDAT('./body_final/exercise_01_12/exercise_02.sas7bdat', encoding='euc-kr') as f:
    for i, row in enumerate(f):
        data.append(row)
        if i == 20000: break
# print(data[0])
# print(data[1])

df02 = pd.DataFrame(columns=data[0])
df02 = df02.append(pd.DataFrame(data[1:], columns=data[0]))
df02.head()
df02.iloc[10]
# columns selection
col02 = []
#%% 
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
datamart00 = pd.read_csv('./body_final/body_datamart_V1_00.csv', skiprows=2).iloc[:-2]
datamart02 = pd.read_csv('./body_final/body_datamart_V1_02.csv', skiprows=5).iloc[:-2]

datamart00_dict = {x:y for x,y in zip(datamart00['변수 이름'], datamart00['레이블'])}
datamart02_dict = {x:y for x,y in zip(datamart02['변수 이름'], datamart02['레이블'])}
#%%
'''
(barycenter template, exercise) pair
'''
# df00 dict
id_sn = []
for i in tqdm(range(len(df00))):
    id_sn.append('_'.join([str(int(x)) for x in df00[['USER_ID_N', 'CNSL_SN_N']].iloc[i].to_list()]))
idsn_df00_dict = {id_sn[i]:df00.iloc[i][4:].to_list() for i in range(len(df00))}

# df02 dict
id_sn = []
for i in tqdm(range(len(df02))):
    id_sn.append('_'.join([str(int(x)) for x in df02[['USER_ID_N', 'CNSL_SN_N']].iloc[i].to_list()]))
idsn_df02_dict = {id_sn[i]:df02.iloc[i][4:].to_list() for i in range(len(df02))}

# body template dictionary
# id_sn = ['_'.join([str(x) for x in body[['USER_ID', 'CNSL_SN']].iloc[i].to_list()]) for i in range(len(body))]
id_sn = []
for i in tqdm(range(len(body))):
    id_sn.append('_'.join([str(x) for x in body[['USER_ID', 'CNSL_SN']].iloc[i].to_list()]))
idsn_template_dict = {x:y for x,y in zip(id_sn, body.sentence)}
len(idsn_template_dict)
#%% matching
i = 2
matched_idsn = list(set(idsn_df00_dict.keys()) & set(idsn_df02_dict.keys()) & set(idsn_template_dict.keys()))
temp00 = ['('+x+'): '+str(y) for x,y in zip(datamart00_dict.values(), idsn_df00_dict.get(matched_idsn[i]))]
temp02 = ['('+x+'): '+str(y) for x,y in zip(datamart02_dict.values(), idsn_df02_dict.get(matched_idsn[i]))]
pair = ['template: ' + idsn_template_dict.get(matched_idsn[i])] + temp00 + temp02
pair
#%%





























