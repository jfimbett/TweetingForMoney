#%%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.corpus import stopwords
import pandas as pd
import io
import os.path
import re
import tarfile
import numpy as np
import time
from multiprocessing import Process, Queue
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import smart_open
import pickle
import time
from tqdm import tqdm
#%%
df = pd.read_csv('tweets_finalv2.csv', low_memory=False)
docs = list(df['text'])

# Tokenize the documents.
# Split the documents into tokens.
tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()  # Convert to lowercase.
    docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

# Remove numbers, but not words that contain numbers.
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

# Remove words that are only one character.
docs = [[token.strip() for token in doc if len(token) > 1] for doc in docs]
# %%
# Lemmatize the documents.
def clean_some_docs(s_docs, i):

    lemmatizer = WordNetLemmatizer()
    s_words = stopwords.words('english')
    s_words = s_words + ['http', 'url', 'tinyurl', 'twurl', 'com', 'html', 'www']

    new_docs = [[] for _ in s_docs] # Pre allocate
    for (i_d, doc) in tqdm(enumerate(s_docs)):
        temp = [lemmatizer.lemmatize(token) for token in doc if not (token in s_words)] 
        new_docs[i_d]=temp
    
    with open(f'docs_{i}', 'wb') as f:
        pickle.dump(new_docs, f)

if __name__== '__main__':
    start=time.time()

    nb=os.cpu_count() 
    batches=np.array_split(docs,nb)
    print(f"Batches of size {[len(batch) for batch in batches]}")

    print(f"Spliting the sample into {len(batches)} batches")
    processes = [Process(target=clean_some_docs, args=(batches[i], i)) for i in range(nb)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    end=time.time()
    time_ellapsed=end-start
    t=round(time_ellapsed, 2)
    n=len(docs)
    print(f"{t} seconds - for {n} tickers")
