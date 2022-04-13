#************************************************************************************************
#************************************************************************************************
# Name: QL_A_Tokenize.py
# Author: Juan Felipe Imbett Jiménez
# Date: 31 October 2017 (Tuesday)
# Description: Parallel version of the cleanTweets routine
#
#
# File was called QL_01_cleanTweets_v2.py
#
#
# LOG From Version 1: 
# TODO:
# Log:
# Version 2 25 October 2017 (Wednesday) 18:23, all the tweets are already in json format, so no need to save anything
# the format use to be like this
# 
# 
#json_object=[{"fullname": "Rupert Meehl", "id": "892397793071050752", "likes": "1", "replies": "0", "retweets": "0",
# "text": "Latest: Trump now at lowest Approval and highest Disapproval ratings yet. Oh, we're winning bigly here ...\n\nhttps://projects.fivethirtyeight.com/trump-approval-ratings/?ex_cid=rrpromo\u00a0\u2026", 
#"timestamp": "2017-08-01T14:53:08", "user": "Rupert_Meehl"}, {"fullname": "Barry Shapiro", "id": "892397794375327744", 
#"likes": "0", "replies": "0", "retweets": "0",
# "text": "A former GOP Rep quoted this line, which pretty much sums up Donald Trump. https://twitter.com/davidfrum/status/863017301595107329\u00a0\u2026", "timestamp": "2017-08-01T14:53:08", "user": "barryshap"}, (...)
#]
#
#They look like this
# {"username": "aamlive", 
#"id": "10023106254",
# "date": "2010-03-05 14:10", 
#"text": "Scott Colyer, CEO Advisors Asset Management featured: Risk-takers\u2019 best bet now? High-yield bonds, says Colyer http://bit.ly/9wozJ8"}
#
# V3: change reading to r not rb 26 October 2017 (Thursday) 18:04
# V4: 26 October 2017 (Thursday) 18:07 change loads() to load when uploading files
# V5: 30 October 2017 (Monday) 13:03 puts element TEST 
# V6: 30 October 2017 (Monday) 17:03 I am having problems saving the json file of external tweets
# 	  my impression is that it is too big to be saved in a text file for json format, I will therefore
# 	  pickle the following files:
#	  tweets_unique_external_tokenized
#     tweets_unique_external_tokenized_grammar
#
#
#
# Log from Version 2
#
# V 2.0 31 October 2017 (Tuesday) 13:07 Parallelization of the code
# V 2.1 01 November 2017 (Wednesday) 15:13 The code took around 13 hours to run, I close the pool
# V 2.2 02 November 2017 (Thursday) 10:44, save tweets in many files to avoid problems when loading
# V 2.3 03 November 2017 (Friday) 11:49 Adds a spelling corrector for words based on the library textblob
# V 2.4 03 November 2017 (Friday) 19:06 Ensures only alphanumeric characters are in each word
# V 2.5 05 November 2017 (Sunday) 15:19 Changes to the methos load and loads, checking the tokenizer and textblob, 
#                                       the regular expression needs to preserve spaces as they are as well as @ symbols or hashtags #
# V 2.6 06 November 2017 (Monday) 15:42 It is taking too much time the split process, will go back and write instead the json file line by line
# V 2.7 07 November 2017 (Tuesday) 07:47 Tokenization splited in two does not make sense, I will do the grammar in the same method, makes it simpler 
#                                        to make it faster
# V 2.8 08 November 2017 (Wednesday) 14:03 Changes in the way external tweets were loaded plus a exception to understand its behavciour
#************************************************************************************************

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

def tokenize_tweet(tweet):
	#print(tweet)
	#print(tweet)
	#print(type(t))
	temp_text=''
	ext=False
	try:
		t=json.loads(tweet)
		temp_text=t["text"]
	except Exception as e:
		#print('Tweet that caused exception ',tweet)
		#print('Interpreted as ', t)
		tweet2=json.dumps(tweet)
		t=json.loads(tweet2)
		try:
			temp_text=t["text"]
		except Exception as e:
			print('Fatal exception ', str(e))
			sys.exit(0)
		#sys.exit(0)

	sentence=onlyAN(temp_text)
	#sentence=TextBlob(sentence)

	# Removes the link, non alphanumeric and spells
	#sentence=sentence.correct()
	#sentence=str(sentence)
	#pos=sentence.find('http')
	#sentence=sentence[:pos]
	words=custom_sent_tokenizer.tokenize(sentence)
	t["tokenized_text"]=words

	# grammar
	tagged=nltk.pos_tag(words)
	t["grammar"]=tagged
	return t

def tokenize_tweet_grammar(tweet):
	words=tweet["tokenized_text"]
	tagged=nltk.pos_tag(words)
	tweet["grammar"]=tagged

	return tweet

def tokenize_tweetv2(temp_text):
    sentence=onlyAN(temp_text)
    words=custom_sent_tokenizer.tokenize(sentence)
    t = {}
    t["tokenized_text"]=words
	# grammar
	tagged=nltk.pos_tag(words)
	t["grammar"]=tagged
    return t

# if __name__ == '__main__':

# 	time_start = time.clock()

# 	Test='' #'' to no test, and 'T' for test

# 	#json_file=open("data\\tweets_unique"+Test+".json", 'r')
	
# 	json_file=open("data\\tweets_unique2017v2.json", 'r')
# 	#json_file=open("data\\mini_test.json", 'r')
# 	data_tweets=json.load(json_file)
# 	# json_file.close()
# 	# json_file=open("data\\tweets_unique_external_final"+Test+".json", 'r')
# 	# #json_file=open("data\\mini_test.json", 'r')
# 	# data_external=json.load(json_file)
# 	# print('Type of external ', type(data_external))
# 	print('Type of internal ', type(data_tweets))
# 	json_file.close()

# 	# Loads object, 
# 	#print('Data uploaded...', len(data_tweets), ' tweets from funds and ', len(data_external), ' tweets from external sources. ')
# 	time_elapsed = (time.clock() - time_start)
# 	print("time elapsed ",time_elapsed/3600," hours")

# 	p=Pool(5)
# 	# print('Tokenizing external tweets')
# 	# data_external1=p.map(tokenize_tweet, data_external)
# 	print('Tokenizing internal tweets')
# 	data_tweets1=p.map(tokenize_tweet, data_tweets)
# 	p.close()

# 	print('Data tokenized')
# 	time_elapsed = (time.clock() - time_elapsed)
# 	print("time elapsed since last checkpoint ",time_elapsed/3600," hours")
# 	print('Saving files')

# 	print("Part of Speech...")
# 	p=Pool(5)
# 	data_tweets_grammar1=p.map(tokenize_tweet_grammar, data_tweets1)
# 	p.close()
# 	#data_external_grammar1=p.map(tokenize_tweet_grammar, data_external1)

# 	#Pickle the list

# 	#Saves json file line by line 
# 	# file=open('data\\tweets_unique_tokenized_grammar2015.json', 'w', encoding="utf-8")
# 	# for tweet in data_tweets_grammar1:
# 	#  	file.write(str(tweet)+"\n")
# 	# file.close()
# 	# file=open('data\\tweets_unique_external_tokenized_grammar'+Test+'.json', 'w')
# 	# for tweet in data_external_grammar1:
# 	# 	file.write(str(tweet)+"\n")
# 	# file.close()
# 	# file=open('data\\pickle\\external_tweets\\list_tweets.pickle', 'wb')
# 	# pickle.dump(data_external1, file)
# 	# file.close()
# 	# print('External saved')
# 	# time_elapsed = (time.clock() - time_elapsed)
# 	print("time elapsed since last checkpoint ",time_elapsed/3600," hours")
# 	file=open('data\\pickle\\tweets\\list_tweets2017v2.pickle', 'wb')
# 	pickle.dump(data_tweets1, file)
# 	file.close()
# 	print('Internal saved')

# 	file=open('data\\pickle\\tweets\\list_tweets_grammar2017v2.pickle', 'wb')
# 	pickle.dump(data_tweets_grammar1, file)
# 	file.close()

# 	print('Internal Grammar Saved')
# 	time_elapsed = (time.clock() - time_elapsed)
# 	print("time elapsed since last checkpoint ",time_elapsed/3600," hours")


# 	# print(" Files with all tweets saved, text has been tokenized, and grammar as Part of Speech has been added")
# 	# print('Total tweets ', len(data_tweets1), ' external ', len(data_external1))
# 	# time_elapsed = (time.clock() - time_elapsed)
# 	# print("time elapsed since last checkpoint ",time_elapsed/3600," hours")
	

