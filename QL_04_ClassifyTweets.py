#************************************************************************************************
#************************************************************************************************
# Name: QL_05_customClassifier.py
# Author: Juan Felipe Imbett Jiménez
# Date: 05 November 2017 (Sunday)
#     This version, 12/10/2021
# Description: Creates a custom classifier based on some ML algorithms, 
# 
# Log:
#************************************************************************************************
#%%
import nltk
import warnings
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from multiprocessing import Process
import time
from tqdm import tqdm

#%% Helps suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
#%%
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers): #List of classifiers
        self._classifiers = classifiers

    def classify(self, features):   #Classifies depending how many algorithms agree
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):  
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf     #Measure of confidence on the classification

#%%
abouts = ['tone', 'topic']
classifiers = [nltk.NaiveBayesClassifier, 
               MultinomialNB(),
               BernoulliNB(),
               LogisticRegression(),
               LinearSVC(),
               SGDClassifier()]
models = []
for about in abouts:
    for classifier in classifiers:
        name = ""
        try:
            name = classifier.__name__
        except:
            name = classifier.__class__.__name__
        open_file = open(f"{name}_{about}.pickle", "rb")
        models.append(pickle.load(open_file))
        open_file.close()




voted_classifier_tone = VoteClassifier(models[0],
                                       models[1],
                                       models[2],
                                       models[3],
                                       models[4],
                                       models[5])

voted_classifier_topic = VoteClassifier(models[6],
                                        models[7],
                                        models[8],
                                        models[9],
                                        models[10],
                                        models[11])
# %%
Nf=4000 #CHECK THAT IS THE SAME OF THE OTHER FILE
word_features = open("word_grammar_features"+str(Nf)+".pickle", "rb")
features = pickle.load(word_features)
word_features.close()

def find_features(grammar): # Grammar is a list of tuples with word and pos_tag
  grammar2=[(x[0], x[1]) for x in grammar]
  word_features = {}
  for w in features:
    word_features[str(w)] = (w in grammar2)
  return word_features

def topic(grammar):
    feats = find_features(grammar)
    return (voted_classifier_topic.classify(feats),voted_classifier_topic.confidence(feats))

def tone(grammar):
    feats = find_features(grammar)
    return (voted_classifier_tone.classify(feats),voted_classifier_tone.confidence(feats))


#%%
def classify_batch(i, ext):

    tweets = []
    with open(f'grammar{ext}{i}', 'rb') as f:
        tweets = pickle.load(f)
    #tweets = tweets[:100]
    ids=[]
    tones = []
    ctones = []
    topics = []
    ctopics=[]
    b = 0 # Batch e.g. each 10,000 tweets 

    k = 0
    for tweet in tqdm(tweets):
        k = k + 1

        if k==len(tweets): #%10000 == 0:
            #print(f"Batch {i} is on subbatch {b}")
            b = b+1
            df = pd.DataFrame.from_dict({'id'     : ids,
                                 'tone'   : tones,
                                 'ctone'  : ctones,
                                 'topic'  : topics, 
                                 'ctopic' : ctopics})

            df.to_csv(f'data/tweets_classified{ext}{i}_{b}.csv', index = False)
            ids=[]
            tones = []
            ctones = []
            topics = []
            ctopics=[]

        ids.append(tweet["id"])
        cl_tones  = tone(tweet['grammar']) 
        cl_topics = topic(tweet['grammar'])
        tones.append(cl_tones[0])
        ctones.append(cl_tones[1])
        topics.append(cl_topics[0])
        ctopics.append(cl_topics[1])



# %%
if __name__ == '__main__':
    
    start=time.time()
    nb=os.cpu_count() 
    ext='_external'
    processes = [Process(target=classify_batch, args=(i,ext)) for i in range(nb)]

    for p in processes:
        p.start()
        
    for p in processes:
        p.join()

    end=time.time()
    time_ellapsed=end-start
    t=round(time_ellapsed, 2)
    print(f"{t} seconds ")

    # Test the data
    # df = pd.DataFrame()
    # for i in range(nb):
    #     df = df.append(pd.read_csv(f"data/tweets_classified{i}_{1}.csv"))
    # print(df.head())
    # print(len(df))
    
