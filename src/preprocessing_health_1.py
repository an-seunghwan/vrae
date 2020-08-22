#%%
'''
- 판단 -> template 생성
- 사람별로 데이터 구분
- 결측치를 고려한 딥러닝 모형 (NLP에서 UNK과 같은 역할)
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
health = pd.read_csv('./data/barycenter_health.csv')
health.head()
health.columns
# 중복된 행 제거
health = health[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
#%% template
template = health.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)
#%%
'''
- key : template dictionary
'''
# body[['USER_ID', 'CNSL_SN']]
user_list = sorted(list(set(health['USER_ID'].to_list())))
len(user_list)
#%% 
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
#%%
'''
<원본 sas 데이터>
- first row is column names
'''
sas_dict = {}
for n, num in enumerate(data_num):
    print('sas data: ', n)
    with SAS7BDAT('./health_final/health_{}.sas7bdat'.format(num), encoding='euc-kr') as f:
        sas = f.to_data_frame()
    sas_dict[n] = sas
#%%
# '''변수 확인'''
# n = 17
# sas_dict[n]
# datamart_ = pd.read_csv('./health_final/health_datamart_V1_{}.csv'.format(data_num[n]), skiprows=0).iloc[:-2]
# [(x,y,z) for x,y,z in zip(datamart_['변수 이름'], datamart_['레이블'], datamart_['유형'])]
#%%
'''
- variable EDA
- missing data 처리 방식: 숫자 = -1 , 문자 = case by case (like dummy coding)
'''
Counter(sas_dict[0]['BLOOD_PRESS'].to_list()) # '131/85'의 형식 -> split lower and upper & 변수 제거
sas_dict[0].insert(list(sas_dict[0].columns).index('BLOOD_PRESS')+1, "BLOOD_PRESS_L", 
                   sas_dict[0]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[1]))
sas_dict[0].insert(list(sas_dict[0].columns).index('BLOOD_PRESS')+2, "BLOOD_PRESS_U", 
                   sas_dict[0]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[0]))
sas_dict[0]['BLOOD_PRESS_L'] = sas_dict[0]['BLOOD_PRESS_L'].map(lambda x : float(x)) 
sas_dict[0]['BLOOD_PRESS_U'] = sas_dict[0]['BLOOD_PRESS_U'].map(lambda x : float(x)) 
#%%
Counter(sas_dict[1]['TARGET_HEART'].to_list()) # '90 ~ 130'의 형식 -> split lower and upper & 변수 제거
sas_dict[1].insert(list(sas_dict[1].columns).index('TARGET_HEART')+1, "TARGET_HEART_L", 
                   sas_dict[1]['TARGET_HEART'].map(lambda x : x.split(' ~ ')).map(lambda x : x[0] if len(x) > 1 else 0))
sas_dict[1].insert(list(sas_dict[1].columns).index('TARGET_HEART')+2, "TARGET_HEART_U", 
                   sas_dict[1]['TARGET_HEART'].map(lambda x : x.split(' ~ ')).map(lambda x : x[1] if len(x) > 1 else 0))
sas_dict[1]['TARGET_HEART_L'] = sas_dict[1]['TARGET_HEART_L'].map(lambda x : float(x)) 
sas_dict[1]['TARGET_HEART_U'] = sas_dict[1]['TARGET_HEART_U'].map(lambda x : float(x)) 

sas_dict[1]['DAY_RECOM_CAL'] = sas_dict[1]['DAY_RECOM_CAL'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거

Counter(sas_dict[1]['TARGET_EX_DAY'].to_list()) # 전부 똑같은 데이터 -> 변수 제거

sas_dict[1]['TARGET_EX_TIME'] = sas_dict[1]['TARGET_EX_TIME'].map(lambda x : float(x)) # as.int

sas_dict[1]['TARGET_WALK'] = sas_dict[1]['TARGET_WALK'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거
#%%
Counter(sas_dict[2]['DAY_ENG_NEED_AM'])
sas_dict[2]['DAY_ENG_NEED_AM'] = sas_dict[2]['DAY_ENG_NEED_AM'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거
#%%
# Counter(sas_dict[15]['BLOOD_PRESS_JUDGE_15'])
# Counter(sas_dict[16]['SUGAR_JUDGE_16'])
#%%
'''변수선택'''
var02 = ['HEIGHT', 'WEIGHT', 'BMI', 'BODY_FAT_PER', 'BLOOD_PRESS_L', 'BLOOD_PRESS_U', 'BLOOD_SUGAR', 'HDL_CHOL', 'NEUTRAL_FAT',
         'WAIST_MSMT', 'MUSCLE', 'BONE_MUSCLE']
var03 = ['RECOM_CAL', 'TARGET_HEART_L', 'TARGET_HEART_U', 'DAY_RECOM_CAL', 'DAY_RECOM_TIME', 'TARGET_EX_TIME', 
         'ACT_VALID_LIM', 'ACT_SAFE_LIM', 'TARGET_WALK']
var04 = ['DAY_ENG_NEED_AM', 'GR_INTAKE_CNT', 'MT_INTAKE_CNT', 'VG_INTAKE_CNT', 'FR_INTAKE_CNT', 'MK_INTAKE_CNT', 
         'WEIGHT_04', 'OBJ_WEIGHT']
# var06 = ['DAY_AVE_ACT_1', 'DAY_AVE_EXCS_1', 'DAY_AVE_ACT_2', 'DAY_AVE_EXCS_2', 'DAY_AVE_ACT_3', 'DAY_AVE_EXCS_3', 'DAY_AVE_ACT_4',
#          'DAY_AVE_EXCS_4', 'DAY_AVE_ACT_5', 'DAY_AVE_EXCS_5', 'DAY_AVE_ACT_6', 'DAY_AVE_EXCS_6', 'DAY_AVE_ACT_7', 'DAY_AVE_EXCS_7']
var07 = ['AVE_WALK_N', 'OBJ_WALK_TOT', 'OBJ_WALK_DAY']
var08 = ['WEIGHT_08']
var09 = ['OBJ_WEIGHT', 'CUR_WEIGHT', 'DIF_WEIGHT']
var10 = ['DAY_ENG_NEED_AM_1', 'CONSUME_CAL_1', 'DAY_ENG_NEED_AM_2', 'CONSUME_CAL_2', 'DAY_ENG_NEED_AM_3', 'CONSUME_CAL_3',
         'DAY_ENG_NEED_AM_4', 'CONSUME_CAL_4', 'DAY_ENG_NEED_AM_5', 'CONSUME_CAL_5', 'DAY_ENG_NEED_AM_6', 'CONSUME_CAL_6', 'DAY_ENG_NEED_AM_7', 'CONSUME_CAL_7']
var11 = ['MEAL_NO', 'TOT_NO', 'AVE_CAL', 'MORNING_CAL', 'LUNCH_CAL', 'DINNER_CAL', 'MORNING_MID_CAL', 'LUNCH_MID_CAL', 'DINNER_MID_CAL']
var12 = ['WEIGHT_12', 'MUSCLE_12', 'BODY_FAT_12', 'BMI_12']
var13 = ['BLOOD_PRESS_MAX', 'BLOOD_PRESS_MIN', 'PULSE']
var14 = ['BLOOD_SUGAR_14']
var15 = ['FOUR_WEEK_AVE_MAX', 'FOUR_WEEK_AVE_MIN', 'EIGHT_WEEK_AVE_MAX', 'EIGHT_WEEK_AVE_MIN', 'TWELVE_WEEK_AVE_MAX', 'TWELVE_WEEK_AVE_MIN']
var16 = ['SUGAR_FOUR_WEEK_AVE', 'SUGAR_EIGHT_WEEK_AVE', 'SUGAR_TWELVE_WEEK_AVE']
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
    scaled_sas = scaler.fit_transform(sas_dict[i][globals()['var{}'.format(num)]].fillna(-1))
    sas_dict_scaled[i][globals()['var{}'.format(num)]] = scaled_sas

'''save sas dict'''
import pickle
with open("./health_final/sas_dict_scaled.pkl", "wb") as f:
    pickle.dump(sas_dict_scaled, f)
#%%