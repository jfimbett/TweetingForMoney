#************************************************************************************************
#************************************************************************************************
# Name: QL_05_mergeData.py
# Author: Juan Felipe Imbett Jiménez
# Date: 05 November 2017 (Sunday)
# Description: Creates a custom classifier based on some ML algorithms, 
# 
# Log:
#************************************************************************************************
#%%
import pandas as pd
import numpy as np
import os
# %%
ns = os.cpu_count()
ext='_external'
df = pd.DataFrame()
for i in range(ns):
    for j in range(1,2):
        df = df.append(pd.read_csv(f"data\\tweets_classified{ext}{i}_{j}.csv", low_memory=False))

df.head()

# %%
tweets = pd.read_csv(f'tweets_unique{ext}.csv', low_memory= False)
tweets = tweets.merge(df, on =['id'])
# %%

from QL_DA01_familyNames import listOfFunds as funds1
from QL_DA01_familyNames import otherFunds as funds2
from QL_DA01_familyNames import external as funds3

funds1=list(set(funds1))
funds2=list(set(funds2))

fund_families=list(set(funds1) | set(funds2))
external=list(set(funds3))

def what_mentions(x):
    mentions=[]
    k = 0
    for fund in fund_families:
        if f"@{fund}" in x:
            mentions.append(fund)
            k = k+1

    return mentions

def many_mentions(x):
    mentions=[]
    k = 0
    for fund in fund_families:
        if f"@{fund}" in x:
            mentions.append(fund)
            k = k+1

    return k

if ext == '_external':
    tweets['mention'] = tweets.text.apply(what_mentions)
    tweets['nmention'] = tweets.text.apply(many_mentions)

tweets.to_csv(f'tweets_final{ext}.csv', index=False)
# %%
