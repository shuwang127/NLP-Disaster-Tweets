rootPath = './'
dataPath = rootPath + '/data/'
tempPath = rootPath + '/temp/'

import os
import re
import sys
import csv
import random
import math
import nltk
import pandas as pd
import numpy as np
from random import choice
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from nltk import word_tokenize
from itertools import chain
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
nltk.download('stopwords')
nltk.download('punkt')

def main():
    # info.
    print("-- AIT726 Project from Julia Jeng, Shu Wang, and Arman Anwar --")
    # load training and testing data.
    dataTrain, dataTest = ReadCsvData()
    # get keywords and vocabulary from training data.
    dList = CreateVocabulary(dataTrain, dataTest)
    # demo
    demo(dataTrain, dataTest, dList, 'Stem', 'TFIDF', 'NaiveBayes')
    return

def demo(dataTrain, dataTest, dList, typeStem, typeFeat, method):
    # input validation.
    if typeStem not in ['NoStem', 'Stem']:
        print('[Error] Stemming setting invalid!')
        return
    if typeFeat not in ['Frequency', 'Binary', 'TFIDF']:
        print('[Error] Feature setting invalid!')
        return
    if method not in ['NaiveBayes', 'Logistic']:
        print('[Error] Classifier setting invalid!')
        return
    print('[Demo] Data: %s | Feature: %s | Classifier: %s.' % (typeStem, typeFeat, method))
    # extract training features and labels.
    ExtractFeatures(dataTrain, dList, 'Train', typeStem, typeFeat)
    # train the model.
    # model = NaiveBayesTrain(featTrain, labelTrain)
    # extract testing features and labels.
    # featTest, labelTest = ExtractFeatures(dataTest, 'test')
    # test the model.
    # predTest = NaiveBayesTest(model, featTest)
    # evaluate.
    # accuracy, confusion = OutputEval(predTest, labelTest)
    # test the data
    return

def ExtractFeatures(dataset, dList, typeSet, typeStem, typeFeat):
    # input validation.
    if typeSet not in ['Train', 'Test']:
        print('[Error] Dataset setting invalid!')
        return
    # sparse the corresponding dataset.
    data = dList['list' + typeSet + typeStem]
    D = len(data)
    # build the vocabulary from training set.
    vocab = list(set(list(chain.from_iterable(dList['listTrain' + typeStem]))))
    V = len(vocab)
    vocabDict = dict(zip(vocab, range(V)))
    print('[Info] Load %d vocabulary words.' % (V))

    label = list(dataset['target'])
    features = np.zeros((D, V))

    # get the feature matrix (Frequency).
    if 'Frequency' == typeFeat:
        for ind, doc in enumerate(data):
            for item in doc:
                if item in vocabDict:
                    features[ind][vocabDict[item]] += 1
    # get the feature matrix (Binary).
    if 'Binary' == typeFeat:
        for ind, doc in enumerate(data):
            for item in doc:
                if item in vocabDict:
                    features[ind][vocabDict[item]] = 1
    # get the feature matrix (TFIDF):
    if 'TFIDF' == typeFeat:
        # get freq and bin features.
        termFreq = np.zeros((D, V))
        termBin = np.zeros((D, V))
        for ind, doc in enumerate(data):
            for item in doc:
                if item in vocabDict:
                    termFreq[ind][vocabDict[item]] += 1
                    termBin[ind][vocabDict[item]] = 1
        # get tf (1+log10)
        tf = np.zeros((D, V))
        for ind in range(D):
            for i in range(V):
                if termFreq[ind][i] > 0:
                    tf[ind][i] = 1 + math.log(termFreq[ind][i], 10)
        del termFreq
        # find idf
        if 'Train' == typeSet:
            # get df
            df = np.zeros((V, 1))
            for ind in range(D):
                for i in range(V):
                    df[i] += termBin[ind][i]
            # get idf (log10(D/df))
            idf = np.zeros((V, 1))
            for i in range(V):
                if df[i] > 0:
                    idf[i] = math.log(D, 10) - math.log(df[i], 10)
            del df
            np.save(tempPath + '/idf_' + typeStem + '.npy', idf)
        else:
            # if 'Test' == dataset, get idf from arguments.
            idf = np.load(tempPath + '/idf_' + typeStem + '.npy')
        del termBin
        # get tfidf
        for ind in range(D):
            for i in range(V):
                features[ind][i] = tf[ind][i] * idf[i]
        np.save(tempPath + '/tfidf_' + typeSet + typeStem + '.npy')

    return features, label

def ReadCsvData():
    # validate temp path
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    # read data from train.csv.
    dataTrain = pd.read_csv(dataPath + 'train.csv')
    print('[Info] Load %d training samples from %s/train.csv.' % (len(dataTrain), dataPath))
    # read data from test.csv.
    dataTest = pd.read_csv(dataPath + 'test_labeled.csv')
    print('[Info] Load %d testing samples from %s/test_labeled.csv.' % (len(dataTest), dataPath))
    # return
    return dataTrain, dataTest

def CreateVocabulary(dataTrain, dataTest):
    # pre-process the data.
    def Preprocess(data):
        # remove url
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        data = re.sub(pattern, '', data)
        # remove html special characters.
        pattern = r'&[(amp)(gt)(lt)]+;'
        data = re.sub(pattern, '', data)
        # remove independent numbers.
        pattern = r' \d+ '
        data = re.sub(pattern, ' ', data)
        # lower case capitalized words.
        pattern = r'([A-Z][a-z]+)'
        def LowerFunc(matched):
            return matched.group(1).lower()
        data = re.sub(pattern, LowerFunc, data)
        # remove hashtags.
        pattern = r'[@#]([A-Za-z]+)'
        data = re.sub(pattern, '', data)
        return data

    # remove stop words.
    def RemoveStop(data):
        dataList = data.split()
        for item in dataList:
            if item.lower() in stopwords.words('english'):
                dataList.remove(item)
        dataNew = " ".join(dataList)
        return dataNew

    # get tokens.
    def GetTokens(data):
        # use tweet tokenizer.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(data)
        tokensNew = []
        # tokenize at each punctuation.
        pattern = r'[A-Za-z]+\'[A-Za-z]+'
        for tk in tokens:
            if re.match(pattern, tk):
                subtokens = word_tokenize(tk)
                tokensNew = tokensNew + subtokens
            else:
                tokensNew.append(tk)
        return tokensNew

    # process tokens with stemming.
    def WithStem(tokens):
        porter = PorterStemmer()
        tokensStem = []
        for tk in tokens:
            tokensStem.append(porter.stem(tk))
        return tokensStem

    # keywords.
    keywdList = list(set(list(dataTrain['keyword'])))
    keywdDict = dict(zip(keywdList, range(len(keywdList))))
    dataTrain['keywd'] = dataTrain['keyword'].apply(lambda x: keywdDict[x])
    dataTest['keywd'] = dataTest['keyword'].apply(lambda x: keywdDict[x])
    # exist location info?
    def is_nan(x):
        return (x is np.nan or x != x)
    dataTrain['loc'] = dataTrain['location'].apply(lambda x: (0 if is_nan(x) else 1))
    dataTest['loc'] = dataTest['location'].apply(lambda x: (0 if is_nan(x) else 1))
    # find url number.
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    dataTrain['url'] = dataTrain['text'].apply(lambda x: len(re.findall(pattern, x)))
    dataTest['url'] = dataTest['text'].apply(lambda x: len(re.findall(pattern, x)))

    # if exist list.npz, load it.
    if os.path.exists(tempPath + 'list.npz'):
        print('[Info] Load text list (noStem/Stem) of train/test set from %s/list.npz.' % (tempPath))
        return np.load(tempPath + 'list.npz', allow_pickle = True)

    # process train list.
    listTrainNoStem = []
    listTrainStem = []
    # read the training data.
    for i in range(len(dataTrain)):
        # get the training data.
        data = dataTrain['text'][i]
        # preprocess the data.
        data = Preprocess(data)
        # remove stop words.
        data = RemoveStop(data)
        # get the tokens for the data.
        tokens = GetTokens(data)
        listTrainNoStem.append(tokens)
        # get the stemmed tokens for the data.
        tokensStem = WithStem(tokens)
        listTrainStem.append(tokensStem)
    # process test list.
    listTestNoStem = []
    listTestStem = []
    # read the testing data.
    for i in range(len(dataTest)):
        # get the testing data.
        data = dataTest['text'][i]
        # preprocess the data.
        data = Preprocess(data)
        # remove stop words.
        data = RemoveStop(data)
        # get the tokens for the data.
        tokens = GetTokens(data)
        listTestNoStem.append(tokens)
        # get the stemmed tokens for the data.
        tokensStem = WithStem(tokens)
        listTestStem.append(tokensStem)
    np.savez(tempPath + 'list.npz', listTrainNoStem=listTrainNoStem, listTrainStem=listTrainStem, listTestNoStem=listTestNoStem, listTestStem=listTestStem)
    print('[Info] Load text list (noStem/Stem) of train/test set from %s/list.npz.' % (tempPath))
    return np.load(tempPath + 'list.npz', allow_pickle = True)

if __name__ == "__main__":
    main()