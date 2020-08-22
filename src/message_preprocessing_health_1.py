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
with SAS7BDAT('./health_final/health_{}.sas7bdat'.format('00'), encoding='euc-kr') as f:
    sas = f.to_data_frame()
#%%
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext.replace('\n', '').replace('\x00', '')
#%%
corpus = [cleanhtml(x) for x in sas['EDITOR_CONT'].to_list() if len(x)]
len(corpus)
okt.pos(corpus[-1])
#%%