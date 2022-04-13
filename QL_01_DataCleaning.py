#********************************************************
# Name: QL_01_DataCleaning.py
# Author: Juan F. Imbet
# Date: 04/10/2021
#
# Description
# At 2021 we decided to update the data of the paper and used a new webscraping code
# to retrieve more data. However, it might occur that some tweets were not in the sample
# if twitter accounts dissapeared. We try to have a unified dataset. 
#********************************************************
#%%
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv
from utils import *
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta

def convert_date(x):
    try:
        return last_day_of_month(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    except:
        try:
            return last_day_of_month(datetime.strptime(f"{x}:01", '%Y-%m-%d %H:%M:%S'))
        except:
            print(f"{x}:01")
            return None

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    ym = next_month - timedelta(days=next_month.day)
    return datetime(int(ym.year), int(ym.month), int(ym.day))


PATH_TO_TWEETS = r"C:\Users\Juan\Documents\TwitterScrapper\tweets"

from QL_DA01_familyNames import listOfFunds as funds1
from QL_DA01_familyNames import otherFunds as funds2
from QL_DA01_familyNames import external as funds3

funds1=list(set(funds1))
funds2=list(set(funds2))

fund_families=list(set(funds1) | set(funds2))
external=list(set(funds3))

# %%
# Loop through all files in the folder
files = list(glob.iglob(f'{PATH_TO_TWEETS}/*.json'))
# Since we dont know ex ante de size, its not very efficient
texts               = []
ids                 = []
usernames           = []
dates               = []
locations           = []
friends_counts      = []
followers_counts    = []
listed_counts       = []
favourites_counts   = []

for filepath in tqdm(files):
    batches = []
    with open(filepath, 'r') as f:
        batches = json.loads(f.read()) 
    for batch in batches:
        for tweet in batch:
            texts.append(tweet["text"])
            ids.append(tweet["id"])
            usernames.append(tweet["username"])
            dates.append(tweet["date"])
            locations.append(tweet["location"])
            friends_counts.append(tweet["friends_count"])
            followers_counts.append(tweet["followers_count"])
            listed_counts.append(tweet["listed_count"])
            favourites_counts.append(tweet["favourites_count"])

info = {"id"               : ids,
        "username"         : usernames, 
        "date"             : dates,
        "text"             : texts, 
        "location"         : locations, 
        "friends_count"    : friends_counts,
        "followers_count"  : followers_counts,
        "listed_count"     : listed_counts,
        "favourites_count" : favourites_counts}
tweets =pd.DataFrame.from_dict(info)

tweets.to_csv("tweets.csv", index = False)
print(len(tweets))

#%% How does this data look like?
tweets['ndate'] = tweets.date.apply(lambda x : str(x).replace("T", " "))
tweets.ndate    = tweets.date.apply(convert_date)
tweets['ym']    = tweets.ndate.apply(last_day_of_month)

# %%
# Can I get also a dataframe of the old tweets?
folders = [f for f in list(glob.glob("../data/*/")) if 'V' in f]
ff = 1
# For debugging purposes we need to make it more verbose, also I will save one file per folder
pbar = tqdm(folders)
for folder in pbar:
    tweets_old = pd.DataFrame()
    fname = folder.split("\\")[-2]
    k = 0
    files = list(glob.glob(f"{folder}\*"))
    if not os.path.exists(f"tweets_old{ff}.csv"):
        for file in files:
            ffname = file.split("\\")[-1]
            pbar.set_description(f"Processing folder {fname} and file {ffname}, {k} files with errors")
            try:
                if 'json' in file:
                    df = pd.DataFrame()
                    with open(file, 'r') as f:
                        info = json.loads(f.read())
                        df = pd.DataFrame.from_dict(info)
                        """
                        {"fullname": "NEIRG", 
                        "id": "504701847165087744", 
                        "likes": "6", 
                        "replies": "0", 
                        "retweets": "0", 
                        "text": "NEINV Strengthens Investment Infrastructure as it Welcomes Les Satlow, CFA, CFP\u00ae #neinv #neinvteam http://tinyurl.com/o526hc8\u00a0", 
                        "timestamp": "2014-08-27T18:47:51", 
                        "user": "_NEIRG_"}
                        """
                        df.rename(columns = {"user" : "username",
                                            "id"   : "id",
                                            "text" : "text",
                                            "timestamp" : "date"})
                        df["location"] = ""
                        df["friends_count"] = np.zeros(len(df))
                        df["followers_count"] = np.zeros(len(df))
                        df["listed_count"] = np.zeros(len(df))
                        df["favourites_count"] = np.zeros(len(df))
                        tweets_old = tweets_old.append(df)
            
                elif 'csv' in file: 
                    try: 
                        df = pd.read_csv(file, sep = ";")
                        df.rename(columns = {"geo" : "location"})
                        if 'username' in df.columns:
                            if np.isnan(df.username[0]):
                                username = file.split("\\")[-1][:-4] # e.g. from '../data\\V5_own\\361Capital.csv' gets 361Capital.csv and then 361Capital 
                    except:
                        df = custom_csv(file)
                    tweets_old = tweets_old.append(df)
            except Exception as err:
                k = k+1
                #print(f"{file} Error: {err}")
        tweets_old.to_csv(f"tweets_old{ff}.csv", index = False)          
    ff +=1

# %%
# Open file by file and check the consistency, I want things like in
# the new file
# Index(['id', 'username', 'date', 'text', 'location', 'friends_count',
#        'followers_count', 'listed_count', 'favourites_count'],
#       dtype='object')
base_columns = pd.read_csv('tweets.csv').columns
df = pd.read_csv("tweets_old1.csv")
#** This one was mostly
df = df.rename(columns = {"id"               : "id",
                          "user"             : "username",
                          "timestamp"        : "date",
                          "text"             : "text",
                          "location"         : "location",
                          "friends_count"    : "friends_count",
                          "followers_count"  : "followers_count",
                          "listed_count"     : "listed_count",
                          "favourites_count" : "favourites_count"})
df.to_csv("tweets_old1.csv")
# ['fullname', 'id', 'likes', 'replies', 'retweets', 'text', 'timestamp',
#        'user', 'location', 'friends_count', 'followers_count', 'listed_count',
#        'favourites_count']

df = pd.read_csv("tweets_old2.csv")

# Index(['id', 'username', 'date', 'text', 'location', 'friends_count',
#        'followers_count', 'listed_count', 'favourites_count'],
#       dtype='object')

df = pd.read_csv("tweets_old3.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old4.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old5.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old6.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old7.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old8.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old9.csv")
assert base_columns.all() == df.columns.all()

df = pd.read_csv("tweets_old10.csv",low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
# Index(['id', 'username', 'date', 'text', 'location', 'friends_count',
#        'followers_count', 'listed_count', 'favourites_count', 'retweets',
#        'favorites', 'geo', 'mentions', 'hashtags', 'permalink'],
#       dtype='object')
assert base_columns.all() == df.columns.all()
df.to_csv("tweets_old10.csv", index = False)

df = pd.read_csv("tweets_old11.csv")
df = df.rename(columns = {"timestamp"        : "date",
                          "user"             : "username"})
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old11.csv", index = False)

df = pd.read_csv("tweets_old12.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old12.csv", index = False)

df = pd.read_csv("tweets_old13.csv", low_memory=False)
df = df.rename(columns = {"timestamp" : "date",
                          "tweet_id"  : "id"})
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
# Index(['fullname', 'html', 'is_retweet', 'likes', 'replies', 'retweet_id',
#        'retweeter_userid', 'retweeter_username', 'retweets', 'text',
#        'timestamp', 'timestamp_epochs', 'tweet_id', 'tweet_url', 'user_id',
#        'username', 'location', 'friends_count', 'followers_count',
#        'listed_count', 'favourites_count'],
#       dtype='object')
df.to_csv("tweets_old13.csv", index = False)

df = pd.read_csv("tweets_old14.csv", low_memory=False)

try:
    df = pd.read_csv("tweets_old15.csv", low_memory=False)
except:
    pass

df = pd.read_csv("tweets_old16.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old16.csv", index = False)

df = pd.read_csv("tweets_old17.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old17.csv", index = False)

df = pd.read_csv("tweets_old18.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old18.csv", index = False)

df = pd.read_csv("tweets_old19.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old19.csv", index = False)

df = pd.read_csv("tweets_old20.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old20.csv", index = False)

df = pd.read_csv("tweets_old21.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old21.csv", index = False)

df = pd.read_csv("tweets_old22.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old22.csv", index = False)

df = pd.read_csv("tweets_old23.csv", low_memory=False)
df = df[['id', 'username', 'date', 'text', 'location', 'friends_count',
       'followers_count', 'listed_count', 'favourites_count']]
df.to_csv("tweets_old23.csv", index = False)

#%%
# Now we append them all
tweets_old = pd.DataFrame()
pbar = tqdm(range(1,24))
for i in pbar:
    try:
        df = pd.read_csv(f"tweets_old{i}.csv", low_memory= False)
        tweets_old = tweets_old.append(df)
    except:
        pbar.set_description(f"File tweets_old{i} not valid")
print(len(tweets_old))
tweets_old.to_csv("tweets_old.csv", index=False)

#%%
def clean_id(x):
    try:
        return int(float(x))
    except:
        return None

tweets_old = pd.read_csv('tweets_old.csv', low_memory=False)
tweets_old.id = tweets_old.id.apply(clean_id)
tweets_old = tweets_old.drop_duplicates(subset=['id'])

tweets_old_external = tweets_old.copy()

tweets_old_external.id = tweets_old_external.id.apply(clean_id)
tweets_old_external = tweets_old_external.drop_duplicates(subset=['id'])
tweets_old_external['external'] = tweets_old_external.username.apply(lambda x : len(list(set(str(x).split(" ")) & set(external))) > 0 )
tweets_old_external = tweets_old_external[tweets_old_external['external']]
tweets_old_external['vintage'] = 2018

tweets_external = pd.read_csv('tweets.csv', low_memory = False)
tweets_external.id = tweets_external.id.apply(clean_id)
tweets_external = tweets_external.drop_duplicates(subset=['id'])
tweets_external['external'] = tweets_external.username.apply(lambda x : len(list(set(str(x).split(" ")) & set(external))) > 0 )
tweets_external = tweets_external[tweets_external['external']]
tweets_external['vintage'] = 2020

#%%
df = tweets_old_external.append(tweets_external)

df['ndate'] = df.date.apply(lambda x : str(x).replace("T", " "))
df.ndate    = df.date.apply(convert_date)
def last_dom(x):
    try:
        return last_day_of_month(x)
    except:
        return np.datetime64("NaT")

df['has_mention'] = df.text.apply(lambda x : '@' in x)
df = df[df['has_mention']]

df['ym']    = df.ndate.apply(last_dom)

df = df.sort_values(['id', 'vintage'])
df = df.groupby(['id']).last()
df = df.reset_index()
df = df.sort_values(['ym'])
ntweets_all = df.groupby('ym').apply(len)
ndates_all  = df.groupby('ym')['ym'].first()

plt.plot(ndates_all, ntweets_all)

texts = df.text


# Check if there is a @
#%%
def mentions_fund(x):
    mentions = False
    for fund in fund_families:
        if f"@{fund}" in x:
            mentions = True
            break
    
    return mentions

df['mentions_fund'] = df.text.apply(lambda x : mentions_fund(x))
df = df[df['mentions_fund']]

df.to_csv('tweets_unique_external.csv', index = False)
#%% # Keep only the ones from the fund families
tweets_old['belong'] = tweets_old.username.apply(lambda x : len(list(set(str(x).split(" ")) & set(fund_families))) > 0 )

tweets_old = tweets_old[tweets_old['belong']]
tweets_old['nmentions'] =  tweets_old.username.apply(lambda x : len(x.split(" ")))
# %%
tweets_old['ndate'] = tweets_old.date.apply(lambda x : str(x).replace("T", " "))
tweets_old.ndate = tweets_old.ndate.apply(convert_date)
tweets_old = tweets_old[~ np.isnat(tweets_old.ndate)]
tweets_old['ym'] = tweets_old.ndate.apply(last_day_of_month)

# %%
ntweets_old = tweets_old.groupby('ym').apply(len)
ndates_old  = tweets_old.groupby('ym')['ym'].first()
plt.plot(ndates_old, ntweets_old)

ntweets = tweets.groupby('ym').apply(len)
ndates  = tweets.groupby('ym')['ym'].first()
plt.plot(ndates, ntweets)

#%% 
tweets['vintage'] = 2021
tweets_old['vintage'] = 2018
# Unique observations?
df = tweets_old.append(tweets)
df = df.sort_values(['id', 'vintage'])
df = df.groupby(['id']).last()
df = df.reset_index()
df = df.sort_values(['ym'])
ntweets_all = df.groupby('ym').apply(len)
ndates_all  = df.groupby('ym')['ym'].first()

plt.plot(ndates_old, ntweets_old)
plt.plot(ndates, ntweets)
plt.plot(ndates_all, ntweets_all)

#%%
df.to_csv("tweets_unique.csv", index = False)
# %%
plt.plot(ndates_all, ntweets_all)
# %%
plt.bar(ndates_all, ntweets_all, width=80)
# %%
