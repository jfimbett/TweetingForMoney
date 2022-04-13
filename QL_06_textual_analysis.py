#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import yfinance as yf
df = pd.read_csv('tweets_final.csv', low_memory=False)


#%%
roots_safe = ['shield',
              'safe', 
              'save',
              'shelter',
              'guard',
              'defend',
              'secure',
              'cauti',
              'pruden',
              'chary',
              'conservative']

roots_trust = ['trust',
'confidence', 'belief', 'faith', 'certainty', 'certitude', 'assurance', 'conviction',
'credence', 'reliance', 'keep', 'protect', 'care', 'trustee', 'guardian']

def safe_word(x):
    is_=False
    for r in roots_safe:
        if r in x:
            is_ = True
            break
    return is_

def trust_word(x):
    is_=False
    for r in roots_trust:
        if r in x:
            is_ = True
            break
    return is_

df['is_safe']=df.text.apply(safe_word)
df['is_trust']=df.text.apply(trust_word)
df.to_csv('tweets_finalv2.csv', index = False)

from datetime import datetime

def str2date(x):
    return datetime.strptime(x, "%Y-%m-%d")

df2 = pd.DataFrame()
df2['mean_safe'] = df.groupby('ym').is_safe.mean()
df2['mean_trust'] = df.groupby('ym').is_trust.mean()
df2.index = df.groupby('ym').ym.first().apply(str2date)

# add data from the SPY


spy = yf.Ticker("SPY")
# get historical market data
hist = spy.history(period="max")



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


hist['date']=hist.index
hist['ym'] = hist.date.apply(convert_date)

df1=pd.DataFrame()
df1['sp500']= hist.groupby('ym').Close.last()

df = pd.merge(df1, df2, left_index=True, right_index=True)

df['ret_sp500']=df.sp500.pct_change()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#df = df[df.index > pd.Timestamp('2010-01-01T12')]
ax1.plot(df.index, df['mean_safe'],  'g-')
ax1.plot(df.index, df['mean_trust'],  'b-')
#ax2.plot(df.index, df['ret_sp500'], 'b-')
ax1.format_xdata = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_minor_locator(mdates.YearLocator(5,month=1,day=1))



# %%
df = pd.read_csv('tweets_finalv2.csv', low_memory=False)


def convert(x):
    if x is None:
        return ""
    else:
        return x

for col in df.columns:
    df[col] = df[col].apply(convert)

def convert_boolean(x):
    if x:
        return 1
    else:
        return 0
#%%
df['belong'] = df['belong'].apply(convert_boolean)

# %%

df = df[['id' , 'text', 'date', 'username', 'timestamp', 'user', 'ym', 'vintage', 'tone', 'ctone', 'topic', 'ctopic', 'is_safe', 'is_trust']]

#%%
df.to_stata('tweets_finalv2.dta', version=118)
# %%
# Dirichlet allocation to Data frame

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
stemmer = SnowballStemmer(language='english')
#nltk.download('wordnet')
# %%
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and (token not in "http"):
            result.append(lemmatize_stemming(token))
    return result
# %%
# Select a text for test
doc_sample = df.text.iloc[0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))
# %%
process_docs = df['text'].map(preprocess)
# %%
dictionary = gensim.corpora.Dictionary(process_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
# %%
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=200)
# %%
bow_corpus = [dictionary.doc2bow(doc) for doc in process_docs]
bow_corpus[4310]
# %%
bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
                                                            bow_doc_4310[i][1]))
# %%
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break
# %%
#%timeit 

lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=4)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
# %%
