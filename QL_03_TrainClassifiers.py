#********************************************************
# Name: QL_03_TrainClassfiers.py
# Author: Juan F. Imbet
# Date: 12/10/2021
#
# Description
# Re-classification of tweets
#********************************************************
#%%

import pandas as pd
import numpy as np
import time
import json
import nltk
from multiprocessing import Pool
import sys
from nltk.tokenize import PunktSentenceTokenizer, TweetTokenizer
from nltk.corpus import twitter_samples
import math
from nltk.corpus import words 
from textblob import Word, TextBlob
import re
from pprint import pprint
import pickle
from multiprocessing import Process
import os
from tqdm import tqdm
import random
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode


#%%

save_word_features = open("training_topic.pickle","rb")
training_topic=pickle.load(save_word_features)
save_word_features.close()


save_word_features = open("training_tone.pickle","rb")
training_tone=pickle.load(save_word_features)
save_word_features.close()

# %%
def find_features(grammar): 
    # Grammar is a list of tuples with word and pos_tag
	grammar2=[(x[0], x[1]) for x in grammar]
	word_features = {}
	for w in features:
		word_features[str(w)] = (w in grammar2)
	return word_features

featuresets_topic = [(find_features(grammar), category) for (grammar, category) in training_topic]
featuresets_tone = [(find_features(grammar), category) for (grammar, category) in training_tone]

# %%
random.seed(55555) #Sets the seed
random.shuffle(featuresets_topic)
random.shuffle(featuresets_tone)
print(len(featuresets_topic), ' for topic and ', len(featuresets_tone), ' for tone')

N_topic=math.floor(len(featuresets_topic)*0.75)
N_tone=math.floor(len(featuresets_tone)*0.75)

testing_set_topic=featuresets_topic[N_topic:]
training_set_topic=featuresets_topic[:N_topic]

testing_set_tone=featuresets_tone[N_tone:]
training_set_tone=featuresets_tone[:N_tone]

#%%
def timing_val(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        return (t2 - t1), res, func.__name__
    return wrapper
#%%
@timing_val
def train_ml(about, classifier):
    training_set = []
    testing_set  = []
    if about=='topic':
        training_set = training_set_topic
        testing_set  = testing_set_topic
    elif about=='tone':
        training_set = training_set_tone
        testing_set  = testing_set_tone
    else:
        pass

    name = ""
    try:
        name = classifier.__name__
    except:
        name = classifier.__class__.__name__

    if name == 'NaiveBayesClassifier':
        model_trained = classifier.train(training_set)
    else:
        model_trained = SklearnClassifier(classifier).train(training_set)

    accuracy = round(nltk.classify.accuracy(model_trained, testing_set)*100,2)
    print(f"{name}_{about}.pickle - Accuracy: {accuracy} %")
    save_classifier = open(f"{name}_{about}.pickle","wb")
    pickle.dump(model_trained, save_classifier)
    save_classifier.close()

abouts = ['tone', 'topic']
classifiers = [nltk.NaiveBayesClassifier, 
               MultinomialNB(),
               BernoulliNB(),
               LogisticRegression(),
               LinearSVC(),
               SGDClassifier()]
for about in abouts:
    for classifier in classifiers:
        train_ml(about, classifier)

