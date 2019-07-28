## file developed by self is giving an error - need to debug

import nltk # Generic NLTK library
# sentence, work tokenizer and unsupervised sentence tokenizer which can be trained and implemented
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer 
# Stemming tool 
from nltk.stem import PorterStemmer
# Text of all the state of union speeches
from nltk.corpus import state_union
# Text of all stop words
from nltk.corpus import stopwords
# Lemmatizer
from nltk.stem import WordNetLemmatizer
# Frequence distribution
from nltk import FreqDist

import numpy as np
import scipy
import matplotlib.pyplot as plt

from nltk.classify.scikitlearn import SklearnClassifier

import random 
import io
import pickle

from nltk.classify import ClassifierI
from statistics import mode

# Writing a class to build an ensemble classifier
class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers): # default method
        self._classifiers=classifiers
        
    def classify(self, features): # returns mode of votes
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features): # returns confidence: fraction of positive votes 
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
            
        choice_votes=votes.count(mode(votes))
        conf =choice_votes/len(votes)
        
        return conf
    
documents_f=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/documents.pickle","rb")
documents=pickle.load(documents_f)
documents_f.close()

word_features5k_f= open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/word_features5k.pickle","rb")
word_features=pickle.load(word_features5k_f)
word_features5k_f.close()

featuresets_f= open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/featuresets.pickle","rb")
featuresets=pickle.load(featuresets_f)
featuresets_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



# building featuresets


random.shuffle(featuresets) #shuffling feature sets for train test

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/MNB_classifier5k.pickle","rb")
MNB_clasifier=pickle.load(open_file)
open_file.close()

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_clasifier=pickle.load(open_file)
open_file.close()

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/LogisticRegression_classifier5k.pickle","rb")
LogisticRegression_clasifier=pickle.load(open_file)
open_file.close()

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/SGDClassifier_classifier5k.pickle","rb")
SGDClassifier_clasifier=pickle.load(open_file)
open_file.close()

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/SVC_classifier5k.pickle","rb")
SVC_clasifier=pickle.load(open_file)
open_file.close()

open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/LinearSVC_classifier5k.pickle","rb")
LinearSVC_clasifier=pickle.load(open_file)
open_file.close()

# open_file=open("/Users/vputcha/Documents_Venkat/Kaggle/NLTK/pickled_algos/NuSVC_classifier5k.pickle","rb")
# NuSVC_classifier=pickle.lead(open_file)
# open_file.close()



voted_classifier = VoteClassifier(
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

# Function to predict sentiment of text and confidence of prediction
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

    