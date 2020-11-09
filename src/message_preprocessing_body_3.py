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
import numpy as np
from tqdm import tqdm
from pprint import pprint
import pickle
from konlpy.tag import Okt
okt = Okt()
from collections import Counter
import re
#%%
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext.replace('\n', '').replace('\x00', '')
#%%
def spacing_tokenize(wrongSentence):
    tagged = okt.pos(wrongSentence)
    
    # spacing
    corrected = ""
    for i in tagged:
        if i[1] in ('Josa', 'PreEomi', 'Eomi', 'Suffix', 'Punctuation'):
            corrected += i[0]
        else:
            corrected += " "+i[0]
    if corrected[0] == " ":
        corrected = corrected[1:]
    
    # tokenize
    temp = corrected.split('다. ')
    tokenized = [okt.morphs(x + '다.') for i,x in enumerate(temp) if i != len(temp)-1] + [temp[-1]]
    
    return tokenized
#%%
with open("./body_final/sas_dict_scaled.pkl", "rb") as f:
    sas_dict = pickle.load(f)
#%%
'''
- barycenter(template) data
'''
body = pd.read_csv('./data/barycenter_body.csv')
body.head()
body.columns
# 중복된 행 제거
body = body[['USER_ID', 'CNSL_SN', 'sentence']].drop_duplicates(['USER_ID', 'CNSL_SN', 'sentence'], keep='first').reset_index(drop=True)
#%%
'''edictor comment and user who has comments'''
with SAS7BDAT('./body_final/exercise_00.sas7bdat', encoding='euc-kr') as f:
    sas = f.to_data_frame()
comment = sas[['USER_ID_N', 'CNSL_SN', 'EDITOR_CONT']]

idx_ = [i for i,x in enumerate(comment['EDITOR_CONT']) if len(x)]
idx_ = list(np.random.choice(idx_, 100, replace=False))
comment = comment.iloc[idx_]
comment['CNSL_SN'] = comment['CNSL_SN'].astype(int)
#%%
'''user id list'''
# user_list1 = sorted(list(set([x.split('_')[0] for x in list(comment_dict.keys())])))
user_list1 = sorted(list(set(comment['USER_ID_N'].to_list())))
user_list2 = sorted(list(set(body['USER_ID'].to_list())))
len(user_list1)
len(user_list2)

user_list_total = sorted(list(set(user_list1).intersection(set(user_list2))))
len(user_list_total)

np.random.seed(1)
idx = np.random.choice(range(len(user_list_total)), int(len(user_list_total) * 0.9), replace=False)
user_list = list(np.array(user_list_total)[idx])
user_val_list = [x for x in user_list_total if x not in user_list]
#%%
'''vocab'''
corpus = []
for i in tqdm(range(len(comment))):
    temp = spacing_tokenize(cleanhtml(comment['EDITOR_CONT'].iloc[i]))
    corpus.extend(temp)
    
vocab = set()
for x in corpus:
    vocab.update(x)
vocab = {x:i+3 for i,x in enumerate(sorted(list(vocab)))}
vocab['PAD'] = 0
vocab['UNK'] = 1
vocab['sos'] = 2
vocab['eos'] = 3
#%%
'''template dictionary'''
template = body.sentence
unique_template = list(set(template.to_list()))
len(unique_template)

template_dict = {x:i+1 for i,x in enumerate(unique_template)}
template_dict['NONE'] = 0

# semantic_template_dict = {}
# for i in range(len(semantic_id)):
#     temp = sorted([x for x in unique_template if semantic_dict.get(x) == i])
#     semantic_template_dict[i] = {x:i+1 for i,x in enumerate(temp)}
#     semantic_template_dict[i][0] = '<NONE>'

# semantic_template_numdict = {}
# for i in range(len(semantic_template_dict)):
#     temp = semantic_template_dict.get(i)
#     semantic_template_numdict[i] = {y:x for x,y in temp.items()}
#%%
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

data_num = ['00', '02', '03', '04', '05', '06', '09', '12']
max_var_num = max(len(globals()['var{}'.format(num)]) for num in data_num)

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
        
        '''message'''
        user_df = pd.merge(comment.loc[np.isin(comment['USER_ID_N'].to_list(), user_key)].reset_index(drop=True), 
                            user_df,
                             how='inner', on='CNSL_SN')
        
        user_df['EDITOR_CONT'] = user_df['EDITOR_CONT'].apply(lambda x:spacing_tokenize(cleanhtml(x)))
        user_df2 = pd.DataFrame()
        for k in range(len(user_df)):
            cont = user_df['EDITOR_CONT'].iloc[k]
            df_ = pd.DataFrame(user_df.iloc[[k], :])
            df_ = pd.DataFrame(np.repeat(df_.values, len(cont), axis=0))
            df_.columns = user_df.columns 
            df_['content'] = [[vocab.get(x) for x in sent] for sent in cont]
            user_df2 = user_df2.append(df_)
        user_df2['template'] = [template_dict.get(x) for x in user_df2['sentence']]
        
        # '''y label'''
        # # user_y = user_df[['USER_ID', 'CNSL_SN', 'sentence']]
        # user_y = user_df['sentence']
        # temp = [semantic_dict.get(x) for x in user_y.to_numpy()]
        # label_y = np.zeros((user_y.shape[0], len(semantic_id)))
        # for i in range(len(temp)):
        #     label_y[i, temp[i]] = semantic_template_dict.get(temp[i]).get(user_y.to_numpy()[i])

        yield user_x, user_x_numeric, user_df2[['USER_ID', 'CNSL_SN', 'content', 'template']]
        # yield user_x, label_y
#%%
user_df_gen = user_df_generator(user_list)
user_x, numeric_x, sentences = next(user_df_gen)


#%%