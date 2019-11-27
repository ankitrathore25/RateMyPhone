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
        # self.test()
        self.readCSVFile()
        # self.preprocess()

    #these vars are used for calculating prior 
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
    #below vars are used to calculate likelihood probability
    classwiseTF = defaultdict(dict) #this dictionary will have only five rows(classes) with terms as col and freq as value

    totalDocument = 0
    allTermsSet = set()
    totalWordsInDataset = 0

    data = {}
    original_dataset = {}
    tf = defaultdict(dict)
    docFreq = defaultdict(dict)
    totalWordsInDoc = {}
    idf = defaultdict()
    allDocIdsList = []
    vocabulary = []
    tfIdfDocumentVector = defaultdict(dict) #this var stores the value of each doc vector(whose magnitude is tf-idf value of each terms)
    tfIdfOfDocumentsForSearchedQuery = defaultdict(dict)

    def get_wordnet_pos(self,pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def dataPruning(self,text):
        if not text:
            return text
        text = text.lower()
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        text = [word for word in text if not any(c.isdigit() for c in word)]
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        text = [t for t in text if len(t) > 0]
        pos_tags = pos_tag(text)
        text = [WordNetLemmatizer().lemmatize(t[0], self.get_wordnet_pos(t[1])) for t in pos_tags]
        text = [t for t in text if len(t) > 1]
        text = " ".join(text)
        return(text)

    def show_wordcloud(self, data, title = None):
        wordcloud = WordCloud(
            background_color = 'white',
            max_words = 600,
            max_font_size = 30, 
            scale = 3,
            random_state = 42
        ).generate(str(data))

        fig = plt.figure(1, figsize = (20, 20))
        plt.axis('off')
        if title: 
            fig.suptitle(title, fontsize = 20)
            fig.subplots_adjust(top = 2.3)

        plt.imshow(wordcloud)
        plt.show()

    def readCSVFile(self) :
        reviews_df = pd.read_csv("Amazon_Unlocked_Mobile.csv")
        # reviews_df = pd.read_csv("testdataP2.csv")
        # reviews_df.head()
        reviews_df = reviews_df[["id","product_name","brand_name","review","rating","review_votes"]]
        # print(reviews_df.head())
        # print(reviews_df)
        print(reviews_df.count())
        print(reviews_df.head())
        reviews_df = reviews_df.drop_duplicates(subset=['brand_name', 'review'], keep='first')
        # print(result_df)
        print(reviews_df.count())
        print(reviews_df.head())
        
        sampled_df = reviews_df.sample(n=30000, random_state=1)
        sampled_df.to_csv("~/Desktop/new_processed_data_30k.csv")
        print(sampled_df.count())
        print(sampled_df.head())

        reviews_df["clean_review"] = reviews_df["review"].apply(lambda x: self.dataPruning(str(x)))
        print(reviews_df.count())
        print(reviews_df.head())

        # self.show_wordcloud(reviews_df["clean_review"])

        countClasss1 = reviews_df['rating'][reviews_df['rating'] == 1].count()
        countClasss2 = reviews_df['rating'][reviews_df['rating'] == 2].count()
        countClasss3 = reviews_df['rating'][reviews_df['rating'] == 3].count()
        countClasss4 = reviews_df['rating'][reviews_df['rating'] == 4].count()
        countClasss5 = reviews_df['rating'][reviews_df['rating'] == 5].count()
        
        print(countClasss1)
        print(countClasss2)
        print(countClasss3)
        print(countClasss4)
        print(countClasss5)

        x = (1,2,3,4,5)
        y = (countClasss1,countClasss2,countClasss3,countClasss4,countClasss5)

        plt.bar(x,y,align='center') # A bar chart
        plt.xlabel('Ratings')
        plt.ylabel('No. of Reviews')
        for i in range(len(y)):
            plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
        plt.show()
    

    


    # if __name__=="__main__":
    #     self.readCSVFile()

    def classify(self,q):
        return ""