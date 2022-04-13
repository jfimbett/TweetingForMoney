#********************************************************
# Name: QL_02_Processtweets.py
# Author: Juan F. Imbet
# Date: 11/10/2021
#
# Description
# Re-classification of tweets
#********************************************************
#%%
import pandas as pd
import numpy as np
import time, json
from multiprocessing import Pool
import sys
from nltk.tokenize import PunktSentenceTokenizer, TweetTokenizer
from nltk.corpus import twitter_samples
import nltk
import math
from nltk.corpus import words 
from textblob import Word, TextBlob
import re
from pprint import pprint
import pickle
from multiprocessing import Process
import time
import os
from tqdm import tqdm
#%%
english_words=words.words()
custom_sent_tokenizer=TweetTokenizer()

def correct_word(word):
	if word in english_words:
		return word
	else:
		#First case has http in the last word
		if 'http' in word:
			pos=word.find('http')
			word_c=word[:pos]
			if word_c in english_words:
				return word_c
			else:
				w=Word(word_c)
				return str(w.correct())
		else:
			#Corrects the word using a spelling corrector
			w=Word(word)
			return str(w.correct())

# Only alphanumeric characters
def onlyAN(word):
	return re.sub(r'[^a-zA-Z0-9 @#]','', word) 

def tokenize_tweets(df, id, ext):
    outfile = open(f"grammar{ext}{id}",'wb')
    
    tt=[]
    for i in tqdm(range(len(df))):
        tweet = df.iloc[i]
        t = {}
        try:
            sentence=onlyAN(tweet.text)
            words=custom_sent_tokenizer.tokenize(sentence)
            t["tokenized_text"] = words
            tagged=nltk.pos_tag(words)
            t["grammar"]=tagged
            t["id"] = tweet.id
        except:
            t={}
        
        tt.append(t)

    pickle.dump(tt,outfile)
    outfile.close()
    

# %%

if __name__ == '__main__':
    ext = '_external'
    start=time.time()
    df = pd.read_csv(f'tweets_unique{ext}.csv', low_memory = False)
    tweets=df[['id', 'text']] #[:100]
    nb=os.cpu_count() 
    batches=np.array_split(tweets,nb)
    print(f"Batches of size {[len(batch) for batch in batches]}")

    print(f"Spliting the sample into {len(batches)} batches")
    processes = [Process(target=tokenize_tweets, args=(batches[i], i, ext)) for i in range(nb)]

    for p in processes:
        p.start()
        

    for p in processes:
        p.join()

    end=time.time()
    time_ellapsed=end-start
    t=round(time_ellapsed, 2)
    n=len(tweets)
    print(f"{t} seconds - for {n} tweets")

    # Open one to check consistency
    tweets = []
    with open('grammar_external1', 'rb') as f:
        tweets = pickle.load(f)
    print(tweets[0])






# %%
