#%%
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

df = pd.read_csv('tweets_finalv2.csv', low_memory=False)


with open('classified_lda', 'rb') as f:
    classified_corpus = pickle.load(f)
# %%
for i in tqdm(range(3)):
    topic = [corpus[i][1] if len(corpus)==3 else 0.0 for corpus in classified_corpus]
    df[f'topic_{i+1}'] = np.array(topic)
  
# %%
df=df[['id', 'topic_1', 'topic_2', 'topic_3']]
# %%
df.to_stata('../tweets_topic.dta')
# %%
