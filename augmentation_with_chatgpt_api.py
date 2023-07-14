import os, re
import time
import openai
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import openpyxl
import math
import tiktoken

import pyarrow.parquet as pq

pd.set_option('mode.chained_assignment',  None) 
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base

openai.api_key = "API_KEY"

db_id = r'ID'
db_pw = r'PW'
db_url = r'URL'
db_port = 'PORT'
db_nm = r'DB_NM'

tmp_str = fr'{db_id}:{db_pw}@{db_url}:{db_port}/{db_nm}'
conn_str = f'mysql://{tmp_str}'
engine = create_engine(f'mysql+pymysql://{tmp_str}')


# @log_time
def read_sql_df() -> pd.DataFrame:
    """
    Fetch target raw data from database
    :return: raw dataframe

    """
    _q = """
        SELECT * 
            FROM DB.TABLE
    """
    # aug_df = pd.read_sql(sql=_q, con=engine)
    aug_df = pd.read_sql(sql=text(_q), con=engine.connect())

    return aug_df

def gen(msg):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=msg
    )

    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens
    print(f'completion_tokens: {output_tokens} \nprompt_tokens: {input_tokens} \ntotal_tokens: {total_tokens}')
    
    return completion['choices'][0]['message']['content'], input_tokens, output_tokens

def c_rsch_summ_eng():
    train = pd.read_csv('/data/nevret/sci_standard_index_translation.csv')
    
    train['check'] = train['c_sm_nm'].apply(lambda x: 1 if x.startswith('달리') else 0)
    train = train[train['check']==0].reset_index(drop=True)
    train = train.drop(columns='check')
    # train = train.iloc[72:]

    aug, input_tokens, output_tokens = [], [], []
    idx, err_cnt = 0, 0
    while True:
        print(idx)
        if idx == len(train):
            break

        text = list(train['c_sm_summ_eng'])[idx]

        messages = [
            {'role': 'user', 
             'content': f'다음 분야와 밀접한 논문 5가지를 각각 제목을 알려주고 내용을 전문용어 10개와 함께 300자 내외로 영문으로 요약해줘: {text}'}
        ]
        
        try:
            aug_text, input_token, output_token = gen(messages)
            aug.append(aug_text)
            input_tokens.append(input_token)
            output_tokens.append(output_token)

            print(aug_text)
            idx += 1

        except:
            print('ERROR!!!!!!')
            err_cnt += 1
            idx += 0
            continue

    df = {'c_sm_summ_eng': aug,
          'c_sm_summ_eng_in_toc': input_tokens,
          'c_sm_summ_eng_out_toc': output_tokens,
        }

    df = pd.DataFrame(df)
    df.to_csv('/data/subclass+eng_1860.csv', index=False, encoding='utf-8-sig')
    print('') 
    
def split_c_rsch_summ_eng():
    # 
    df = pd.read_csv('/data/nevret/subclass_paper.csv')

    train = read_sql_df()
    train['check'] = train['c_sm_nm'].apply(lambda x: 1 if x.startswith('달리') else 0)
    train = train[train['check']==0].reset_index(drop=True)
    train = train.drop(columns='check')

    sentence_list = []
    for i in range(len(df)):
        sentence_list.append(df['c_sm_paper_eng'][i].split('\n\n'))

    tmp = pd.DataFrame()
    for i in sentence_list:
        tmp = pd.concat([tmp, pd.Series(i)], axis=1)

    result = pd.concat([train[['c_sm_summ_eng']], tmp.T.reset_index(drop=True)], axis=1)
    print('')

    return result

def process_c_rsch_summ_eng():
    df = pd.read_csv('/data/paper_4preproc_postproc.csv')
    df['4'] = df['4'].astype(str)

    # text = df['c_sm_summ_eng'][722]
    # messages = [
    # {'role': 'user', 
    #     'content': f'다음 분야와 밀접한 논문 5가지를 각각 제목을 알려주고 내용을 영어로 요약해줘: {text}'}
    # ]
    # aug_text, input_token, output_token = gen(messages)
    # print(aug_text)

    for idx in range(len(df)):
        if not df['4'][idx].startswith('5'):
        # if df['3'][idx] is np.nan:
            df.iloc[idx, 1:] = np.nan
        
    idx, err_cnt = 0, 0
    while True:
        print(idx)
        if idx == len(df):
            break
        elif not df['0'][idx] is np.nan:
            idx += 1
            continue

        text = list(df['c_sm_summ_eng'])[idx]

        messages = [
            {'role': 'user', 
             'content': f'다음 분야와 밀접한 논문 5가지를 각각 제목을 알려주고 내용을 영어로 요약해줘: {text}'}
        ]

        try:
            aug_text, input_token, output_token = gen(messages)

            if len(aug_text.split('\n\n')):
                df['0'][idx] = aug_text.split('\n\n')[0]
                df['1'][idx] = aug_text.split('\n\n')[1]
                df['2'][idx] = aug_text.split('\n\n')[2]
                df['3'][idx] = aug_text.split('\n\n')[3]
                df['4'][idx] = aug_text.split('\n\n')[4]

            if len(aug_text.split('\n\n')) > 5:
                df['5'][idx] = aug_text.split('\n\n')[5]

            print(aug_text)
            idx += 1

        except:
            print('ERROR!!!!!!')
            err_cnt += 1
            idx += 0
            continue
        
    print('')

# sci_standard_index_rsch()
def sci_standard_index_rsch():
    subclass_df = read_sql_df()    # limenet.sci_standard_index1
    subclass_df = subclass_df.dropna(subset=['c_sm_summ_eng'])
    subclass_df = subclass_df[['cat_id', '1st_paper', '2nd_paper', '3rd_paper', '4th_paper', '5th_paper']]
    
    tmp_df = pd.melt(subclass_df, id_vars=['cat_id'], value_vars=['1st_paper', '2nd_paper', '3rd_paper', '4th_paper', '5th_paper']).sort_values(by=['cat_id', 'variable'])
    tmp_df = tmp_df[['cat_id', 'value']].rename(columns={'value': 'rsch_summ'}).reset_index(drop=True)

    conn = engine.connect()
    tmp_df.to_sql(name='sci_standard_index_rsch', con=engine, if_exists='append', index=False)
    conn.close()

# STEP 4
def augmentation_sci_standard_index_rsch_appl():
    rsch_df = pd.read_csv('/data/nevret/sci_standard_index_rsch.csv')
    rsch_df = rsch_df[['cat_id', 'value']].rename(columns={'value': 'rsch_summ'}).reset_index(drop=True)
    
    org_df = read_sql_df()
    org_df = org_df[['cat_id', 'c_sm_nm_eng']]
    
    rsch_df = rsch_df.merge(org_df, on='cat_id', how='left')

    rsch_df = rsch_df.iloc[4000:8000, :].reset_index(drop=True)

    aug, input_tokens, output_tokens = [], [], []
    idx, err_cnt = 0, 0
    while True:
        print(idx)
        if idx == len(rsch_df):
            break

        text = list(rsch_df['rsch_summ'])[idx]
        rsch = list(rsch_df['c_sm_nm_eng'])[idx]

        messages = [
            {'role': 'user', 
            #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
             'content': f'Summarize 10 ways to apply the following research findings in at least 400 words each, numbered and including jargon closely related to {rsch}: {text}'}
        ]

        try:
            start = time.time()
            aug_text, input_token, output_token = gen(messages)

            aug.append(aug_text)
            input_tokens.append(input_token)
            output_tokens.append(output_token)

            print(aug_text)
            idx += 1
            
            end = time.time()
            print(f"{end - start:.5f} sec")

        except:
            print('ERROR!!!!!!')
            err_cnt += 1
            idx += 0
            continue

    df['aug_text'] = pd.Series(aug)
    df.to_csv('tmptmp1.csv', index=False, encoding='utf-8-sig')
    
def postprocess_sci_standard_index_rsch_appl():
    ANOMALY = False
    CONCAT = False

    if CONCAT: 
        # main()
        # split_aug()
        # postprocess_aug()
        # last_postproc()
        # step4()
        # df = pd.read_csv('/data/paper_final.csv')
        df = pd.read_csv('/data/nevret/aug0-352.csv')
        df1 = pd.read_csv('/data/nevret/legal_bert/tmptmp1.csv')
        df2 = pd.read_csv('/data/nevret/legal_bert/tmptmp2.csv')
        df3 = pd.read_csv('/data/nevret/legal_bert/tmptmp3.csv')
        df4 = pd.read_csv('/data/nevret/legal_bert/tmptmp4.csv')
        df5 = pd.read_csv('/data/nevret/legal_bert/tmptmp5.csv')
        df6 = pd.read_csv('/data/nevret/legal_bert/tmptmp6.csv')

        df = pd.concat([df, df1, df2, df3, df4, df5, df6], axis=0).reset_index(drop=True)

        rsch_df = pd.read_csv('/data/nevret/sci_standard_index_rsch.csv')
        rsch_df = rsch_df[['cat_id', 'value']].rename(columns={'value': 'rsch_summ'}).reset_index(drop=True)
        
        org_df = read_sql_df()
        org_df = org_df[['cat_id', 'c_sm_nm_eng']]
        
        rsch_df = rsch_df.merge(org_df, on='cat_id', how='left')

        # rsch_df = rsch_df.iloc[:353, :].reset_index(drop=True)
        rsch_df['index'] = rsch_df.index
        rsch_df['number'] = rsch_df['cat_id'].astype(str) + '-' + rsch_df['index'].astype(str)

        ### Generate one sample data
        if ANOMALY:
            text = list(rsch_df['rsch_summ'])[343]
            rsch = list(rsch_df['c_sm_nm_eng'])[343]
            
            messages = [
                {'role': 'user', 
                #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
                'content': f'Summarize 10 ways to apply the following research findings in at least 400 words each, numbered and including jargon closely related to {rsch}: {text}'}
            ]

            aug_text, input_token, output_token = gen(messages)

            df['aug_text'][343] = aug_text
            df['aug_text_input_tokens'][343] = input_token
            df['aug_text_output_tokens'][343] = output_token
            df.to_csv('/data/nevret/aug0-352.csv', index=False)


        df['aug_list'] = df['aug_text'].apply(lambda x: x.split('\n\n'))
        df['len'] = df['aug_list'].apply(lambda x: len(x))

        max_len = df['len'].max()

        # make empty max len columns
        for i in range(max_len):
            df[f'{i}_aug_list'] = pd.Series()

        # N filtering
        for _ in range(10):
            for i in df['aug_list']:
                if i == [] or i[0].startswith('1.') or i[0].startswith('1)'):
                    continue
                else:
                    del i[0]

        # filtering length
        df['len'] = df['aug_list'].apply(lambda x: len(x))

        # aug data split append
        for i in range(len(df)):
            for j in range(df['len'][i]):
                df[f'{j}_aug_list'][i] = df['aug_list'][i][j]

        df = df.fillna('')
        for i in range(len(df['9_aug_list'])):
            if df['9_aug_list'][i].startswith('10.') or df['9_aug_list'][i].startswith('10)'):
                continue
            else:
                df.loc[i] = ''


    #### PROCESS 2
    rsch_df = pd.read_csv('sci_standard_index_rsch.csv')
    rsch_df = rsch_df[['cat_id', 'value']].rename(columns={'value': 'rsch_summ'}).reset_index(drop=True)
    
    org_df = pd.read_csv('sci_standard_index_translation1.csv')
    org_df = org_df[['cat_id', 'c_sm_nm_eng']]
    
    rsch_df = rsch_df.merge(org_df, on='cat_id', how='left')

    # rsch_df = rsch_df.iloc[:353, :].reset_index(drop=True)
    rsch_df['index'] = rsch_df.index
    rsch_df['number'] = rsch_df['cat_id'].astype(str) + '-' + rsch_df['index'].astype(str)

    df = pd.read_csv('aug_process1_df.csv')
    df = df.fillna('')

    # 비어있는 index 추출 후, 재생성
    empty_index = [i for i in range(len(df)) if df['aug_list'][i] == '']

    if ANOMALY:
        idx = 0
        while True:
            if idx == len(empty_index):
                break

            text = list(rsch_df['rsch_summ'])[empty_index[idx]]
            rsch = list(rsch_df['c_sm_nm_eng'])[empty_index[idx]]
            
            messages = [
                {'role': 'user', 
                #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
                'content': f'Summarize 10 ways to apply the following research findings in at least 400 words each, numbered and including jargon closely related to {rsch}: {text}'}
            ]

            try:
                aug_text, input_token, output_token = gen(messages)
                print(idx)
                print(aug_text)

            except:
                print(idx)
                print('@@@@@@@@@@@@@@@ ERROR @@@@@@@@@@@@@@@')
                idx += 0
                continue

            df['aug_text'][empty_index[idx]] = aug_text
            df['aug_text_input_tokens'][empty_index[idx]] = input_token
            df['aug_text_output_tokens'][empty_index[idx]] = output_token
            
            idx += 1

def postprocess_sci_standard_index_rsch_appl2():
    df = pd.read_excel('jargon3.xlsx')

    rsch_df = pd.read_csv('sci_standard_index_rsch.csv')
    rsch_df = rsch_df[['cat_id', 'value']].rename(columns={'value': 'rsch_summ'}).reset_index(drop=True)
    
    org_df = pd.read_csv('sci_standard_index_translation1.csv')
    # org_df = org_df[['cat_id', 'c_sm_nm_eng']]
    
    rsch_df = rsch_df.merge(org_df[['cat_id', 'c_sm_nm_eng']], on='cat_id', how='left')

    # rsch_df = rsch_df.iloc[:353, :].reset_index(drop=True)
    rsch_df['index'] = rsch_df.index
    rsch_df['number'] = rsch_df['cat_id'].astype(str) + '-' + rsch_df['index'].astype(str)

    df['aug_list'] = df['aug_text'].apply(lambda x: x.split('\n\n'))
    df['len'] = df['aug_list'].apply(lambda x: len(x))

    max_len = df['len'].max()

    # make empty max len columns
    for i in range(max_len):
        df[f'{i}_aug_list'] = pd.Series()

    # N filtering
    for _ in range(10):
        for i in df['aug_list']:
            if i == [] or i[0].startswith('1.') or i[0].startswith('1)'):
                continue
            else:
                del i[0]

    # filtering length
    df['len'] = df['aug_list'].apply(lambda x: len(x))

    # aug data split append
    for i in range(len(df)):
        for j in range(df['len'][i]):
            df[f'{j}_aug_list'][i] = df['aug_list'][i][j]

    df = df.fillna('')
    for i in range(len(df['9_aug_list'])):
        if df['9_aug_list'][i].startswith('10.') or df['9_aug_list'][i].startswith('10)'):
            continue
        else:
            df.loc[i] = ''

    # 비어있는 index 추출 후, 재생성
    empty_index = [i for i in range(len(df)) if df['aug_list'][i] == '']

    if empty_index != []:
        idx = 0
        while True:
            if idx == len(empty_index):
                break

            text = list(rsch_df['rsch_summ'])[empty_index[idx]]
            rsch = list(rsch_df['c_sm_nm_eng'])[empty_index[idx]]
            
            messages = [
                {'role': 'user', 
                #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
                'content': f'Summarize 10 ways to apply the following research findings in at least 400 words each, numbered and including jargon closely related to {rsch}: {text}'}
            ]

            try:
                aug_text, input_token, output_token = gen(messages)
                print(idx)
                print(aug_text)

            except:
                print(idx)
                print('@@@@@@@@@@@@@@@ ERROR @@@@@@@@@@@@@@@')
                idx += 0
                continue

            df['aug_text'][empty_index[idx]] = aug_text
            df['aug_text_input_tokens'][empty_index[idx]] = input_token
            df['aug_text_output_tokens'][empty_index[idx]] = output_token
            
            idx += 1

    df = df.iloc[:, :15]
    # df = df.drop(columns=['aug_list', 'len'])
    # df.to_csv('result_jargon.csv', index=False, encoding='utf-8-sig')

    concat_df = pd.concat([rsch_df, df], axis=1)
    concat_df = pd.melt(concat_df, id_vars=['cat_id', 'number'], value_vars=['0_aug_list', '1_aug_list', '2_aug_list', '3_aug_list', '4_aug_list', '5_aug_list', '6_aug_list', '7_aug_list', '8_aug_list', '9_aug_list']).sort_values(by=['cat_id', 'number', 'variable']).reset_index(drop=True)
    # concat_df = concat_df[['cat_id', 'value']]

    rsch_id = pd.Series([i for i in range(1, 6)] * (len(concat_df) // 5))
    app_id = pd.Series([i for i in range(1, 11)] * (len(concat_df) // 10))
    concat_df['value'] = concat_df['value'].apply(lambda x: re.sub(r'^[\d\W]+. ', '', x))

    concat_df['number'] = rsch_id
    concat_df['variable'] = app_id

    concat_df = concat_df.rename(columns={'number': 'rsch_id', 
                                          'variable': 'app_id',
                                          'value': 'app_text'})
    
    concat_df.to_csv('concat_jargon.csv', index=False, encoding='utf-8-sig')
    
    conn = engine.connect()
    concat_df.to_sql(name='sci_standard_index_rsch_appl', con=engine, if_exists='append', index=False)
    conn.close()

    print('')

def make_translation_sci_standard_index_rsch_appl():
    df = pd.read_csv('concat_jargon.csv')
    df['app_translation_text'] = pd.Series(dtype='float64')
    df['app_text_input_tokens'] = pd.Series(dtype='float64')
    df['app_text_output_tokens'] = pd.Series(dtype='float64')

    # 1 / 4
    df = df.iloc[:30000, :]

    idx = 0
    while True:
        if idx == len(df):
            break

        text = df['app_text'][idx]
        messages = [
            {'role': 'user', 
            #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
            'content': f'Translate the following sentence into Korean: {text}'}
        ]

        try:
            start = time.time()
            aug_text, input_token, output_token = gen(messages)
            print(idx)
            print(aug_text)
            end = time.time()
            print(f"{end - start:.5f} sec")

        except:
            print(idx)
            print('@@@@@@@@@@@@@@@ ERROR @@@@@@@@@@@@@@@')
            idx += 0
            continue

        df['app_translation_text'][idx] = aug_text
        df['app_text_input_tokens'][idx] = input_token
        df['app_text_output_tokens'][idx] = output_token
        
        idx += 1

    df.to_csv('translation1.csv', index=False, encoding='utf-8-sig')
    print('')

def concat_translation_sci_standard_index_rsch_appl():
    df1 = pd.read_csv('translation1.csv', encoding='utf-8-sig')
    df2 = pd.read_csv('translation2.csv', encoding='utf-8-sig')
    df3 = pd.read_csv('translation3.csv', encoding='utf-8-sig')
    df4 = pd.read_csv('translation4.csv', encoding='utf-8-sig')

    # OUTPUT TOKENS
    tokenizer = tiktoken.get_encoding("cl100k_base")  # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # tokenizer = tiktoken.get_encoding("gpt-3") 
    # tokenizer = tiktoken.encoding_for_model("gpt2")

    output_tokens_lengths = []
    tk0 = tqdm(df4['app_translation_text'].fillna('').values, total=len(df4))
    for text in tk0:
        length = len(tokenizer.encode(text))
        output_tokens_lengths.append(length)
    
    df4['app_text_output_tokens'] = output_tokens_lengths

    df = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)
    df['app_text_input_tokens'] = df['app_text_input_tokens'].astype(int)
    df['app_text_output_tokens'] = df['app_text_output_tokens'].astype(int)

    df.to_csv('app_translation.csv', index=False, encoding='utf-8-sig')

    conn = engine.connect()
    df.to_sql(name='sci_standard_index_rsch_appl', con=engine, if_exists='replace', index=False)    # if_exists='replace' 덮어씌우기
    conn.close()

    print('')

def chatgpt_aug_test():
    df = pd.read_csv('app_translation.csv', encoding='utf-8-sig')
    df = df.iloc[:100, :]

    df['augmentation_text'] = pd.Series(dtype='float64')
    df['aug_text_input_tokens'] = pd.Series(dtype='float64')
    df['aug_text_output_tokens'] = pd.Series(dtype='float64')

    idx = 0
    while True:
        if idx == len(df):
            break

        text = df['app_translation_text'][idx]
        messages = [
            {'role': 'user', 
            #  'content': f'다음 연구내용을 응용하는 방법 10가지를 {rsch}와 밀접한 전문용어들을 포함하여 숫자를 붙여서 각 300자 이상 영어로 요약해줘: {text}'}
            'content': f'Please tell us in Korean about 250 words each of three research studies that can be applied in relation to the following specialized fields: {text}'}
        ]

        try:
            start = time.time()
            aug_text, input_token, output_token = gen(messages)
            print(idx)
            print(aug_text)
            end = time.time()
            print(f"{end - start:.5f} sec")

        except:
            print(idx)
            print('@@@@@@@@@@@@@@@ ERROR @@@@@@@@@@@@@@@')
            idx += 0
            continue

        df['augmentation_text'][idx] = aug_text
        df['aug_text_input_tokens'][idx] = input_token
        df['aug_text_output_tokens'][idx] = output_token
        
        idx += 1

    df.to_csv('translation1.csv', index=False, encoding='utf-8-sig')
    print('')
    print('')

def calculate_token_length():
    # OUTPUT TOKENS
    tokenizer = tiktoken.get_encoding("cl100k_base")  # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # tokenizer = tiktoken.get_encoding("gpt-3") 
    # tokenizer = tiktoken.encoding_for_model("gpt2")

    text = 'Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens.'
    # text = 'Multiple models each with different capabilities and price points Prices are per 1000 tokens You can think of tokens as pieces of words where 1000 tokens is about 750 words'

    output_tokens_lengths = []
    length = len(tokenizer.encode(text))
    output_tokens_lengths.append(length)

    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc)

    print([tokenizer.decode([i]) for i in enc])
    # text = df['app_translation_text'][idx]
    messages = [
        {'role': 'user', 
         'content': f'다음 문단을 그대로 출력해줘: {text}'}
        # 'content': f'Please tell us in Korean about 250 words each of three research studies that can be applied in relation to the following specialized fields: {text}'}
    ]

    start = time.time()
    aug_text, input_token, output_token = gen(messages)
    print(aug_text)


    print('')
    
if __name__ == '__main__':
    chatgpt_aug_test()
