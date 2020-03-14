rootPath = './'
dataPath = rootPath + '/data/'
tempPath = rootPath + '/temp/'

import os
import re
import random
import math
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
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
    # demo NaiveBayes
    #demo(dataTrain, dataTest, dList, 'NoStem', 'Frequency', 'NaiveBayes')
    #demo(dataTrain, dataTest, dList, 'NoStem', 'Binary', 'NaiveBayes')
    #demo(dataTrain, dataTest, dList, 'NoStem', 'TFIDF', 'NaiveBayes')
    #demo(dataTrain, dataTest, dList, 'Stem', 'Frequency', 'NaiveBayes')
    #demo(dataTrain, dataTest, dList, 'Stem', 'Binary', 'NaiveBayes')
    #demo(dataTrain, dataTest, dList, 'Stem', 'TFIDF', 'NaiveBayes')
    # demo Logistic
    demo(dataTrain, dataTest, dList, 'NoStem', 'Frequency', 'Logistic')
    #demo(dataTrain, dataTest, dList, 'NoStem', 'Binary', 'Logistic')
    #demo(dataTrain, dataTest, dList, 'NoStem', 'TFIDF', 'Logistic')
    #demo(dataTrain, dataTest, dList, 'Stem', 'Frequency', 'Logistic')
    #demo(dataTrain, dataTest, dList, 'Stem', 'Binary', 'Logistic')
    #demo(dataTrain, dataTest, dList, 'Stem', 'TFIDF', 'Logistic')
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
    print('[Demo] ------ Data: %s | Feature: %s | Classifier: %s ------' % (typeStem, typeFeat, method))
    # extract training features and labels.
    featTrain, labelTrain = ExtractFeatures(dataTrain, dList, 'Train', typeStem, typeFeat)
    # extract testing features and labels.
    featTest, labelTest = ExtractFeatures(dataTest, dList, 'Test', typeStem, typeFeat)
    # train and test the model.
    if 'NaiveBayes' == method:
        prior, likelihood = NaiveBayesTrain(featTrain, labelTrain)
        predTest = NaiveBayesTest(prior, likelihood, featTest)
    elif 'Logistic' == method:
        model = LogisticTrain(featTrain, labelTrain, featTest, labelTest)
        predTest = LogisticTest(model, featTest)
    # evaluate.
    accuracy, confusion = OutputEval(predTest, labelTest, typeStem, typeFeat, method)
    return

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
    print('[Info] Load %d \'%s\' vocabulary words.' % (V, typeStem))

    # get labels.
    labels = np.array(dataset['target']).reshape(-1, 1)
    # get text features.
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
        if os.path.exists(tempPath + '/tfidf_' + typeSet + typeStem + '.npy'):
            features = np.load(tempPath + '/tfidf_' + typeSet + typeStem + '.npy')
        else:
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
            if os.path.exists(tempPath + '/idf_' + typeStem + '.npy'):
                idf = np.load(tempPath + '/idf_' + typeStem + '.npy')
            elif 'Train' == typeSet:
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
                print('[Error] Need file: %s/idf_%s.npy to process test data!' % (tempPath, typeStem))
                return
            del termBin
            # get tfidf
            for ind in range(D):
                for i in range(V):
                    features[ind][i] = tf[ind][i] * idf[i]
            np.save(tempPath + '/tfidf_' + typeSet + typeStem + '.npy', features)

    # mix features.
    # loc? (1 or 0)
    featloc = np.array(dataset['loc']).reshape(-1, 1)
    features = np.hstack((features, featloc))
    # url number.
    featurl = np.array(dataset['url']).reshape(-1, 1)
    features = np.hstack((features, featurl))
    # keywd onehot (222).
    featkeywd = np.array(dataset['keywd']).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse = False)
    featkeywd = onehot_encoder.fit_transform(featkeywd)
    features = np.hstack((features, featkeywd))
    # return features and labels
    print('[Info] Get %d \'%s\' %sing features (dim:%d) and labels (dim:1).' % (len(features), typeFeat, typeSet.lower(), len(features[0])))
    return features, labels

def NaiveBayesTrain(featTrain, labelTrain):
    # define the log prior.
    def GetLogPrior(labelTrain):
        # count the number.
        nDoc = len(labelTrain)
        nPos = list(labelTrain).count(1)
        nNag = list(labelTrain).count(0)
        # calculate the logprior.
        priorPos = math.log(nPos / nDoc)
        priorNag = math.log(nNag / nDoc)
        prior = [priorNag, priorPos]
        return prior

    # define loglikelihood.
    def GetLogLikelihood(features, labels):
        # get V and D.
        V = len(features[0])
        D = len(features)
        cls = 2
        # initilaze likelihood matrix.
        likelihood = np.zeros((cls, V))
        for ind in range(D):
            for i in range(V):
                likelihood[labels[ind][0]][i] += features[ind][i]
        # Laplace smoothing.
        denom = np.zeros((cls, 1))
        for lb in range(cls):
            denom[lb] = sum(likelihood[lb]) + V
            for i in range(V):
                likelihood[lb][i] += 1
                likelihood[lb][i] /= denom[lb]
                likelihood[lb][i] = math.log(likelihood[lb][i])
        return likelihood

    # get the log prior.
    prior = GetLogPrior(labelTrain)
    # get the log likelihood
    likelihood = GetLogLikelihood(featTrain, labelTrain)
    print('[Info] Naive Bayes classifier training done!')
    return prior, likelihood

def NaiveBayesTest(prior, likelihood, featTest):
    # get V and D.
    V = len(featTest[0])
    D = len(featTest)
    cls = 2
    # get pred(D, cls) matrix and predictions(D, 1).
    pred = np.zeros((D, cls))
    predictions = np.zeros((D, 1))
    for ind in range(D):
        for lb in range(cls):
            pred[ind][lb] += prior[lb]
            for i in range(V):
                pred[ind][lb] += likelihood[lb][i] * featTest[ind][i]
        predictions[ind] = list(pred[ind]).index(max(pred[ind]))
    print('[Info] Naive Bayes classifier testing done!')
    return predictions

class LogisticRegression(nn.Module):
    def __init__(self, dims):
        super(LogisticRegression, self).__init__()
        self.L1 = nn.Linear(dims, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.sigmoid(self.L1(x))
        return a1

def LogisticTrain(featTrain, labelTrain, featTest, labelTest, rate = 0.1, iternum = 10000, chknum = 100):
    # initialize network weights with uniform distribution.
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight)
            nn.init.uniform_(m.bias)

    # get vector dimension and train/test number.
    dims = len(featTrain[0])
    numTrain = len(featTrain)
    numTest = len(featTest)

    # shuffle the data and label.
    index = [i for i in range(numTrain)]
    random.shuffle(index)
    featTrain = featTrain[index]
    labelTrain = labelTrain[index]
    index = [i for i in range(numTest)]
    random.shuffle(index)
    featTest = featTest[index]
    labelTest = labelTest[index]

    # convert data (x,y) into tensor.
    xTrain = torch.Tensor(featTrain).cuda()
    yTrain = torch.LongTensor(labelTrain).cuda()
    xTest = torch.Tensor(featTest).cuda()
    yTest = torch.LongTensor(labelTest).cuda()

    # convert to mini-batch form.
    batchsize = 256
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size = batchsize, shuffle = False)
    test = torchdata.TensorDataset(xTest, yTest)
    testloader = torchdata.DataLoader(test, batch_size = batchsize, shuffle = False)

    # build the model of feed forward neural network.
    print('[Para] Learning Rate = %.2f, Iteration Number = %d.' % (rate, iternum))
    model = LogisticRegression(dims)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.apply(weight_init)
    model.to(device)
    # optimizing with stochastic gradient descent.
    optimizer = optim.SGD(model.parameters(), lr = rate)
    # seting loss function as mean squared error.
    criterion = nn.MSELoss()

    # run on each epoch.
    accList = [0]
    for epoch in range(iternum):
        # training phase.
        model.train()
        lossTrain = 0
        accTrain = 0
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(label.float(), yhat)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item()
            preds = (yhat > 0.5).long()
            accTrain += torch.sum(torch.eq(preds, label).long()).item()
        lossTrain /= (iter + 1)
        accTrain *= 100 / numTrain

        # testing phase.
        model.eval()
        accTest = 0
        with torch.no_grad():
            for iter, (data, label) in enumerate(testloader):
                data = data.to(device)
                label = label.to(device)
                yhat = model.forward(data)  # get output
                # statistic
                preds = (yhat > 0.5).long()
                accTest += torch.sum(torch.eq(preds, label).long()).item()
        accTest *= 100 / numTest
        accList.append(accTest)

        # output information.
        if 0 == (epoch + 1) % chknum:
            print('[Epoch %03d] Loss: %.3f, TrainAcc: %.3f%%, TestAcc: %.3f%%' % (epoch + 1, lossTrain, accTrain, accTest))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + '/model_logistic.pth')
        # stop judgement.
        if (epoch + 1) >= chknum and accList[-1] < min(accList[-chknum:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model_logistic.pth'))
    print('[Info] Logistic Regression classifier training done!')
    return model

def LogisticTest(model, featTest):
    D = len(featTest)
    x = torch.Tensor(featTest).cuda()
    yhat = model.forward(x)
    predictions = np.zeros((D, 1))
    for ind in range(D):
        if yhat[ind] > 0.5:
            predictions[ind][0] = 1
    print('[Info] Logistic Regression classifier testing done!')
    return predictions

def OutputEval(predictions, labels, typeStem, typeFeat, method):
    # evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    def Evaluation(predictions, labels):
        D = len(labels)
        cls = 2
        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind][0])
            nCol = int(labels[ind][0])
            confusion[nRow][nCol] += 1
        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D
        return accuracy, confusion

    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions, labels)
    # output on screen and to file.
    print('       -------------------------------------------')
    print('       ' + typeStem + ' | ' + typeFeat + ' | ' + method)
    print('       accuracy : %.2f%%' % (accuracy * 100))
    print('       confusion matrix :      (actual)')
    print('                           Neg         Pos')
    print('       (predicted) Neg     %-4d(TN)    %-4d(FN)' % (confusion[0][0], confusion[0][1]))
    print('                   Pos     %-4d(FP)    %-4d(TP)' % (confusion[1][0], confusion[1][1]))
    print('       -------------------------------------------')
    return accuracy, confusion

if __name__ == "__main__":
    main()