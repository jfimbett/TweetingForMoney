# %%
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.corpus import stopwords
import pandas as pd
import io
import os.path
import re
import tarfile
from tqdm import tqdm
from gensim.models import Phrases
import os
import pickle
# Remove rare and common tokens.
from gensim.corpora import Dictionary
# Train LDA model.
from gensim.models import LdaModel, LdaMulticore

docs = []
for i in range(os.cpu_count()):
    with open(f'docs_{i}', 'rb') as f:
        temp = pickle.load(f)
    docs = docs+temp

#%%
# Quick check, some stopwords are continuing
print("Checking stopwords that were not filtered")
sw = ['http', 'url', 'tinyurl', 'twurl', 'com', 'html', 'www', 'bit', 'ly', 'amp', 'u']
for (i_d,doc) in tqdm(enumerate(docs)):
    temp = doc
    for token in doc:
        if token in sw:
            temp.remove(token)
    docs[i_d] = temp


#%%
# Compute bigrams
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in tqdm(range(len(docs))):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
# %%


# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents
dictionary.filter_extremes(no_below=20)
# %%
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]
# %%
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
# %%


# Set training parameters.
num_topics = 3
chunksize = 20000
passes = 20
iterations = 100
workers = 6
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

if __name__ == '__main__':
    model = LdaMulticore(
        workers = workers,
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    # %%
    top_topics = model.top_topics(corpus) #, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)
    # %%
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    for i_t, t in enumerate(top_topics):
        freqs = t[0]
        info = {}
        for f in freqs:
            info[f[1]] = f[0]

        plt.figure()
        plt.imshow(WordCloud(background_color='white').fit_words(info))
        plt.axis("off")
        plt.title("Topic #" + str(i_t))
        plt.show()
    # %%
    # Classify the new tweets
    classified_corpus = [model[doc] for doc in corpus]
    print(classified_corpus[:20])

    with open('classified_lda', 'wb') as f:
        pickle.dump(classified_corpus, f)