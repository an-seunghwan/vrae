#%%
'''
- 판단 -> template 생성
- 사람별로 데이터 구분
- 결측치를 고려한 딥러닝 모형 (NLP에서 UNK과 같은 역할)
*****
수많은 한글 변수 -> 어떻게 처리?
*****
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
nutri = pd.read_csv('./data/barycenter_nutrition.csv')
nutri.head()
nutri.columns
# 중복된 행 제거
nutri = nutri[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
#%% template
template = nutri.sentence
uniqe_template = list(set(template.to_list()))
len(uniqe_template)
#%%
'''
- key : template dictionary
'''
# body[['USER_ID', 'CNSL_SN']]
user_list = sorted(list(set(nutri['USER_ID'].to_list())))
len(user_list)
#%% 
'''
<데이터 마트>
- 변수명을 한글로 보기 위함(가독성)
'''
data_num = ['00', '02', '03', '05', '06', '07', '08',
            '11', '12', '13', '14', '15', '16']
datamart_dict = {}
for n, num in enumerate(data_num):
    datamart = pd.read_csv('./nutrition_final/nutrition_datamart_V1_{}.csv'.format(num), skiprows=0).iloc[:-2]
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
    with SAS7BDAT('./nutrition_final/nutrient_{}.sas7bdat'.format(num), encoding='euc-kr') as f:
        sas = f.to_data_frame()
    sas_dict[n] = sas
#%%
'''변수 확인'''
n = 14
sas_dict[n]
datamart_ = pd.read_csv('./nutrition_final/nutrition_datamart_V1_{}.csv'.format(data_num[n]), skiprows=0).iloc[:-2]
[(x,y,z) for x,y,z in zip(datamart_['변수 이름'], datamart_['레이블'], datamart_['유형'])]
#%%
'''
- variable EDA
- missing data 처리 방식: 숫자 = -1 , 문자 = case by case (like dummy coding)
'''
# dummy variables
Counter(sas_dict[0]['PA'])
dict_ = {'': 0, 'M100' : 1, 'M125' : 2, 'M111' : 3, 'M148': 4, 'F100' : 5, 'F112' : 6, 'F127': 7, 'F145': 8}
sas_dict[0]['PA'] = sas_dict[0]['PA'].map(lambda x : dict_.get(x)) # dummy

Counter(sas_dict[0]['WEIGHT_CTRL_PER'])
dict_ = {'': 0, 'M005' : 1, 'M010' : 2, 'M015' : 3, 'M020': 4, 'P000' : 5, 'P005' : 6, 'P015': 7, 'P020': 8}
sas_dict[0]['WEIGHT_CTRL_PER'] = sas_dict[0]['WEIGHT_CTRL_PER'].map(lambda x : dict_.get(x)) # dummy

Counter(sas_dict[0]['FRST_MEAL_CRCTN'])
dict_ = {'': 0, 'IN01' : 1, 'IN02' : 2, 'IN03' : 3, 'SE01': 4, 'SE02' : 5, 'SE03' : 6, 'SE04': 7, 'SE05': 8,
         'SE06': 9, 'SE07': 10} 
sas_dict[0]['FRST_MEAL_CRCTN'] = sas_dict[0]['FRST_MEAL_CRCTN'].map(lambda x : dict_.get(x)) # dummy

Counter(sas_dict[0]['SCND_MEAL_CRCTN'])
dict_ = {'': 0, 'IN01' : 1, 'IN02' : 2, 'IN03' : 3, 'SE01': 4, 'SE02' : 5, 'SE03' : 6, 'SE04': 7, 'SE05': 8,
         'SE06': 9, 'SE07': 10} 
sas_dict[0]['SCND_MEAL_CRCTN'] = sas_dict[0]['SCND_MEAL_CRCTN'].map(lambda x : dict_.get(x)) # dummys
#%%
Counter(sas_dict[1]['BLOOD_PRESS'].to_list()) # '131/85'의 형식 -> split lower and upper & 변수 제거
sas_dict[1].insert(list(sas_dict[1].columns).index('BLOOD_PRESS')+1, "BLOOD_PRESS_L", 
                   sas_dict[1]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[1]))
sas_dict[1].insert(list(sas_dict[1].columns).index('BLOOD_PRESS')+2, "BLOOD_PRESS_U", 
                   sas_dict[1]['BLOOD_PRESS'].map(lambda x : x.split('/')).map(lambda x : x[0]))
sas_dict[1]['BLOOD_PRESS_L'] = sas_dict[1]['BLOOD_PRESS_L'].map(lambda x : float(x)) 
sas_dict[1]['BLOOD_PRESS_U'] = sas_dict[1]['BLOOD_PRESS_U'].map(lambda x : float(x)) 
#%%
Counter(sas_dict[2]['RECOM_CAL_03'])
sas_dict[2]['RECOM_CAL_03'] = sas_dict[2]['RECOM_CAL_03'].map(lambda x : float(re.sub(',', '', x))) # ,(comma) 제거

Counter(sas_dict[2]['BODY_ACT_COEF_03'])
dict_ = {'': 0, 'M100' : 1, 'M125' : 2, 'M111' : 3, 'M148': 4, 'F100' : 5, 'F112' : 6, 'F127': 7, 'F145': 8}
sas_dict[2]['BODY_ACT_COEF_03'] = sas_dict[2]['BODY_ACT_COEF_03'].map(lambda x : dict_.get(x)) # dummy

Counter(sas_dict[2]['WEIGHT_CONT_RATIO_03'])
dict_ = {'': 0, 'M005' : 1, 'M010' : 2, 'M015' : 3, 'M020': 4, 'P000' : 5, 'P005' : 6, 'P015': 7, 'P020': 8}
sas_dict[2]['WEIGHT_CONT_RATIO_03'] = sas_dict[2]['WEIGHT_CONT_RATIO_03'].map(lambda x : dict_.get(x)) # dummy
#%%
Counter(sas_dict[10]['SORT'])

Counter(sas_dict[10]['MENU']) # word2vec

Counter(sas_dict[10]['AMOUNT']) # word2vec

Counter(sas_dict[10]['NOTE']) # word2vec

Counter(sas_dict[10]['MEAL_PLACE']) 
dict_ = {'집': 0, '외식' : 1}
sas_dict[10]['MEAL_PLACE'] = sas_dict[10]['MEAL_PLACE'].map(lambda x : dict_.get(x)) # dummy
#%%

'''13번 sas data = word2vec'''

#%%
Counter(sas_dict[13]['MEAL_CD']) # word2vec
dict_ = {'10': 0, '20' : 1}
sas_dict[13]['MEAL_CD'] = sas_dict[13]['MEAL_CD'].map(lambda x : dict_.get(x)) # dummy

Counter(sas_dict[13]['MEAL_NM'])
dict_ = {'아침': 0, '점심' : 1}
sas_dict[13]['MEAL_NM'] = sas_dict[13]['MEAL_NM'].map(lambda x : dict_.get(x)) # dummys
#%%
'''변수선택'''
var00 = ['PA', 'WEIGHT_CTRL_PER', 'DAY_ENG_NEED_AM', 'CAL_DAY_ENG_NEED_AM', 'OBJ_DAY_ENG_NEED_AM', 'CAL_OBJ_DAY_ENG_NEED_AM',
        'FRST_MEAL_CRCTN', 'SCND_MEAL_CRCTN', 'GR_INTAKE_CNT', 'MT_INTAKE_CNT', 'VG_INTAKE_CNT', 'FR_INTAKE_CNT', 'MK_INTAKE_CNT',
        'WEIGHT', 'OBJ_WEIGHT', 'CARB_RECOM_PER', 'PROTEIN_RECOM_PER', 'FAT_RECOM_PER']
var02 = ['HEIGHT', 'WEIGHT', 'BMI', 'BODY_FAT_PER', 'BLOOD_PRESS_L', 'BLOOD_PRESS_U', 'BLOOD_SUGAR', 'HDL_CHOL', 'NEUTRAL_FAT',
         'WAIST_MSMT', 'MUSCLE', 'BONE_MUSCLE']
var03 = ['RECOM_CAL_03', 'GR_INTAKE_CNT_03', 'MT_INTAKE_CNT_03', 'VG_INTAKE_CNT_03', 'FR_INTAKE_CNT_03', 'MK_INTAKE_CNT_03',
         'BODY_ACT_COEF_03', 'WEIGHT_CONT_RATIO_03', 'DAY_ENG_NEED_AM_03', 'CAL_DAY_ENG_NEED_AM_03', 'TAGET_DAY_ENG_NEED_AM_03', 'CT_DAY_ENG_NEED_AM_03',
        'WEIGHT_03', 'TARGET_WEIGHT_03']
var05 = ['GR_INTAKE_CNT_05', 'MT_INTAKE_CNT_05', 'VG_INTAKE_CNT_05', 'FR_INTAKE_CNT_05', 'MK_INTAKE_CNT_05', 'REG_NUM_05',
        'TOT_CAL_05', 'AVE_CAL_05', 'ACT_MISSION_RESP_RATIO_05', 'ACT_MISSION_SUCC_RATIO_05']
var06 = ['WEEK_DAY', 'MORNING_CAL', 'LUNCH_CAL', 'DINNER_CAL', 'MORNING_MID_CAL', 'LUNCH_MID_CAL', 'DINNER_MID_CAL']
var07 = ['USER_ID_N', 'USER_ID', 'WEEK_DAY', 'CARB_08', 'FAT_08', 'PROTEIN_08']
var08 = ['CARB_09', 'PROTEIN_09', 'FAT_09', 'SUGARS_09', 'SALT_09', 'CHOLESTEROL_09', 'SATURATED_09', 'TRANS_09', 'CALCIUM_09']
var11 = ['SORT', 'MENU', 'AMOUNT', 'NOTE', 'GR_INTAKE_CNT_05', 'MT_INTAKE_CNT_05', 'VG_INTAKE_CNT_05', 'FR_INTAKE_CNT_05', 'MK_INTAKE_CNT_05']
var12 = ['TOT_CAL', 'MORNING_CAL', 'LUNCH_CAL', 'DINNER_CAL', 'MID_CAL']
var13 = 
var14 = ['MEAL_CD', 'MEAL_NM', 'CAL']
var15 = ['CARB_15', 'PROTEIN_15', 'FAT_15', 'TOTAL_15']
var16 = ['CAL_16', 'CARB_16', 'PROTEIN_16', 'FAT_16', 'SUGARS_16', 'SALT_16', 'CHOLESTEROL_16', 'SATURATED_16', 'TRANS_16', 'CALCIUM_16']
#%%
