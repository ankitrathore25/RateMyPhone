from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import defaultdict
from wordcloud import WordCloud
from numpy import dot
from numpy.linalg import norm
from markupsafe import Markup
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

import matplotlib.pyplot as plt
import random

import nltk
import os
import string
import numpy as np
# import copy
import pandas as pd
# import pickle
# import re
import math
import csv
# import copy
import operator

class naiveBayesClassifier():
    def __init__(self):
        print("In implemented naive bayes method.")
        # print("commented reviewNBClassifier preprocessing")
        self.readCSVFile()
        self.preprocess()

    countClasss1 = 0
    countClasss2 = 0
    countClasss3 = 0
    countClasss4 = 0
    countClasss5 = 0
    class1TF = {}
    class2TF = {}
    class3TF = {}
    class4TF = {}
    class5TF = {}
    split_ratio = 0.95
    #below vars are used to calculate likelihood probability
    classwiseTF = defaultdict(dict) #this dictionary will have only five rows(classes) with terms as col and freq as value

    train_data = []
    test_data = []
    allTermsSet = set()

    def readCSVFile(self) :
        print("reading data for classifier from csv file")
        reviews_df = pd.read_csv("dataset/Amazon_Unlocked_Mobile_5k.csv")
        # reviews_df = pd.read_csv("testdataP2.csv")
        print("Total data size: "+str(reviews_df.shape[0]))
        reviews_df['split'] = np.random.randn(reviews_df.shape[0], 1)
        msk = np.random.rand(len(reviews_df)) <= self.split_ratio #data split
        self.train_data = reviews_df[msk]
        self.test_data = reviews_df[~msk]
        print("Train data size: "+str(self.train_data.shape[0]))
        print("Test data size: "+str(self.test_data.shape[0]))
        
        # with open('Amazon_Unlocked_Mobile.csv') as csvfile:
        #     readCSV = csv.reader(csvfile, delimiter=",")
        #     for row in readCSV:
        #         fileData = row[4]
        #         if len(fileData.strip()) > 0:
        #             self.data.append(row)
        # print("in classifier readCsvFile: "+str(len(self.data)))

    def remove_stop_words(self, document):
        if not document:
            return ""
        stop_words = stopwords.words('english')
        words = word_tokenize(str(document))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w.strip()
        return new_text.strip()


    def stemming(self, document):
        if not document:
            return ""
        stemmer= PorterStemmer()
        # stemmer = SnowballStemmer("english")
        # stemmer = LancasterStemmer()
        tokens = word_tokenize(str(document))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    def lemmatize(self, document):
        if not document:
            return ""
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(str(document))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + lemmatizer.lemmatize(w,pos="v")
        return new_text

    def docuDataPruning(self, document):
        if not document:
            return ""
        else:
            document = self.remove_stop_words(document)
            document = self.lemmatize(document)
            document = self.stemming(document)
            if not document:
                return ""
            else:
                return document.lower().strip()

    def preprocess(self):
        #self.totalDocument
        #self.data
        for i in range(len(self.train_data)):
            # row = self.train_data.iloc[[i]]
            # print(row)
            # print(self.train_data.iloc[i]["review"])
            # print(self.train_data.iloc[i]["rating"])
            # print(row.loc[])
            review_tokens = set(word_tokenize(self.docuDataPruning(self.train_data.iloc[i]["review"])))
            rating = self.train_data.iloc[i]["rating"]
            if rating == 1:
                self.countClasss1 += 1
                for term in review_tokens:
                    self.allTermsSet.add(term)
                    if term in self.class1TF:
                        self.class1TF[term] += 1
                    else:
                        self.class1TF[term] = 1
            elif rating == 2:
                self.countClasss2 += 1
                for term in review_tokens:
                    self.allTermsSet.add(term)
                    if term in self.class2TF:
                        self.class2TF[term] += 1
                    else:
                        self.class2TF[term] = 1
            elif rating == 3:
                self.countClasss3 += 1
                for term in review_tokens:
                    self.allTermsSet.add(term)
                    if term in self.class3TF:
                        self.class3TF[term] += 1
                    else:
                        self.class3TF[term] = 1
            elif rating == 4:
                self.countClasss4 += 1
                for term in review_tokens:
                    self.allTermsSet.add(term)
                    if term in self.class4TF:
                        self.class4TF[term] += 1
                    else:
                        self.class4TF[term] = 1
            elif rating == 5:
                self.countClasss5 += 1
                for term in review_tokens:
                    self.allTermsSet.add(term)
                    if term in self.class5TF:
                        self.class5TF[term] += 1
                    else:
                        self.class5TF[term] = 1
        print("preprocessing done")
 

    def calculate_cond_prob_for_class(self, term, classs):
        cond_prob = 0
        if classs == 1:
            if self.countClasss1 != 0:
                if term in self.class1TF:
                    cond_prob = (self.class1TF[term]+1)/(self.countClasss1+1)
                else:
                    cond_prob = 1/(self.countClasss1+1)
        elif classs == 2:
            if self.countClasss2 != 0:
                if term in self.class2TF:
                    cond_prob = (self.class2TF[term]+1)/(self.countClasss2+1)
                else:
                    cond_prob = 1/(self.countClasss2+1)
        elif classs == 3:
            if self.countClasss3 != 0:
                if term in self.class3TF:
                    cond_prob = (self.class3TF[term]+1)/(self.countClasss3+1)
                else:
                    cond_prob = 1/(self.countClasss3+1)
        elif classs == 4:
            if self.countClasss4 != 0:
                if term in self.class4TF:
                    cond_prob = (self.class4TF[term]+1)/(self.countClasss4+1)
                else:
                    cond_prob = 1/(self.countClasss4+1)
        elif classs == 5:
            if self.countClasss5 != 0:
                if term in self.class5TF:
                    cond_prob = (self.class5TF[term]+1)/(self.countClasss5+1)
                else:
                    cond_prob = 1/(self.countClasss5+1)
        return cond_prob
    
    def classify(self,q):
        if not q:
            return ""
        query_tokens = word_tokenize(self.docuDataPruning(q))
        conditional_prob = {}
        for i in range(5):
            term_cond_prob = 1
            for term in query_tokens:
                term_cond_prob *= self.calculate_cond_prob_for_class(term, i+1)
            conditional_prob[i] = term_cond_prob
        
        classifyResult = {}
        classifyResult["rating"] = max(conditional_prob.keys(), key=(lambda key: conditional_prob[key])) + 1
        classifyResult["allClassProbability"] = conditional_prob
        return classifyResult
        # return max(conditional_prob.keys(), key=(lambda key: conditional_prob[key]))
        # return conditional_prob

    def evaluate_test_data(self):
        # confusion_mat = defaultdict(dict)
        train_data_size = self.train_data.shape[0]
        test_data_size = self.test_data.shape[0]
        total_data_size = train_data_size + test_data_size
        print("Total data size: "+str(total_data_size))
        print("Train data size: "+str(train_data_size))
        print("Test data size: "+str(test_data_size))
        print("Split ratio: "+str(self.split_ratio))
        #predicted\Original
        #  1 2 3 4 5
        #1
        #2
        #3
        #4
        #5
        w = 5
        h = 5
        matrix = [[0 for x in range(w)] for y in range(h)] 
        for i in range(len(self.test_data)):

            review = self.docuDataPruning(self.test_data.iloc[i]["review"])
            rating = self.test_data.iloc[i]["rating"] - 1 #1 is subtracted to match column in mat
            predicted_rating = self.classify(review) 
            if str(rating) and str(predicted_rating):
                matrix[int(predicted_rating)][int(rating)] += 1
        
        for k in range(5):
            print(matrix[k])

        sumPredictedClassCount = []
        for i in range(5):
            sum1 = 0
            for j in range(5):
                sum1 += matrix[i][j]
            sumPredictedClassCount.append(sum1)
        #precision of every class
        P1 = matrix[0][0]/sumPredictedClassCount[0]
        P2 = matrix[1][1]/sumPredictedClassCount[1]
        P3 = matrix[2][2]/sumPredictedClassCount[2]
        P4 = matrix[3][3]/sumPredictedClassCount[3]
        P5 = matrix[4][4]/sumPredictedClassCount[4]

        sumOriginalClassCount = []
        for i in range(5):
            sum2 = 0
            for j in range(5):
                sum2 += matrix[j][i]
            sumOriginalClassCount.append(sum2)
        #recall of every class
        R1 = matrix[0][0]/sumOriginalClassCount[0] 
        R2 = matrix[1][1]/sumOriginalClassCount[1]
        R3 = matrix[2][2]/sumOriginalClassCount[2]
        R4 = matrix[3][3]/sumOriginalClassCount[3]
        R5 = matrix[4][4]/sumOriginalClassCount[4]
        
        OverallPrecision = (P1 + P2 + P3 + P4 + P5)/5
        OverallRecall = (R1 + R2 + R3 + R4 + R5)/5
        print("P1: "+str(P1)+", P2: "+str(P2)+", P3: "+str(P3)+", P4: "+str(P4)+", P5: "+str(P5))
        print("R1: "+str(R1)+", R2: "+str(R2)+", R3: "+str(R3)+", R4: "+str(R4)+", R5: "+str(R5))
        print("Overall Precision: "+str(OverallPrecision))
        print("Overall Recall: "+str(OverallRecall))

        FPClasswiseCount = []
        FNClasswiseCount = []
        TNClasswiseCount = []
        TPClasswiseCount = []
        w = 0
        for w in range(5):
            TPClasswiseCount.append(matrix[w][w])
            FNClasswiseCount.append(sumOriginalClassCount[w] - matrix[w][w])
            FPClasswiseCount.append(sumPredictedClassCount[w] - matrix[w][w])
            TNClasswiseCount.append(sum(matrix[0])+sum(matrix[1])+sum(matrix[2])+sum(matrix[3])+sum(matrix[4]) - FNClasswiseCount[w] - FPClasswiseCount[w] + matrix[w][w])
            
        for k in range(5):              
            print("Class "+str(k+1)+" - TPR: "+str(TPClasswiseCount[k]/(sumOriginalClassCount[k]))+", FPR: "+str(FPClasswiseCount[k]/(total_data_size-sumOriginalClassCount[k])))
        
        modelF1Score = (2*OverallPrecision*OverallRecall)/(OverallPrecision + OverallRecall)

        print("F1 score: "+str(modelF1Score))

        accuracy = []
        for k1 in range(5):
            accuracy.append((TPClasswiseCount[k1]+TNClasswiseCount[k1])/test_data_size)
        print("ACC1: "+str(accuracy[0])+", ACC2: "+str(accuracy[1])+", ACC3: "+str(accuracy[2])+", ACC4: "+str(accuracy[3])+", ACC5: "+str(accuracy[4]))        

        print("Overall Accuracy: "+ str(sum(accuracy)/5))




            # review = self.docuDataPruning(self.test_data.iloc[i]["review"])
            # rating = self.train_data.iloc[i]["rating"] - 1 #1 is subtracted to match column in mat
            # classified = self.classify(review)
            # predicted_rating = max(classified.values()) - 1 #1 is subtracted to match column in mat
            # if predicted_rating in confusion_mat:
            #     if rating in confusion_mat[predicted_rating]:
            #         confusion_mat[predicted_rating][rating] += 1
            #     else:
            #         confusion_mat[predicted_rating] = {str(rating):1}
            # else:
            #     val = {str(rating):1}
            #     confusion_mat = {str(predicted_rating):val}


    # if __name__=="__main__":
    #     firstfunctionToBeCalled()
    #     callSearch("life learning")


    
