from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
from markupsafe import Markup
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

import nltk
import os
import string
import math
import csv
import operator

class imageMain():
    def __init__(self):
        # print("commented preprocessing for search engine")
        self.readCSVFile()
        self.preprocess()

    data = {}
    original_dataset = {}
    tf = defaultdict(dict)
    docFreq = defaultdict(dict)
    totalDocument = 0
    totalWordsInDoc = {}
    idf = defaultdict()
    allDocIdsList = []
    allTermsSet = set()
    tfIdfDocumentVector = defaultdict(dict) #this var stores the value of each doc vector(whose magnitude is tf-idf value of each terms)
    tfIdfOfDocumentsForSearchedQuery = defaultdict(dict)

    def readCSVFile(self) :
        print("reading images csv")
        dataset = {}
        self.original_dataset
        with open('dataset/images.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            #[0:'Id', 1:'url', 2:'caption']
            for row in readCSV:
                self.original_dataset[row[0]] = row
                if len(row[2].strip()) > 0:
                    dataset.update({row[0]: row[2].strip()})
        self.data = dataset
        print("in image captioning readCsvFile: "+str(len(self.data)))

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
                return document.lower()

    def preprocess(self):
        for (docId, document) in self.data.items():
            self.allDocIdsList.append(docId)
            self.totalDocument = self.totalDocument + 1
            self.data[docId] = self.docuDataPruning(document).strip()
            terms = word_tokenize(self.data[docId])
            self.totalWordsInDoc[docId] = len(terms)
            for term in terms:
                self.allTermsSet.add(term)
                if term in self.tf:
                    row = self.tf[term]
                    if docId in row:
                        row[docId] = (row[docId] + 1)
                    else:
                        row[docId] = 1
                else:
                    self.tf[term] = {str(docId):1}
                if term in self.docFreq:
                    self.docFreq[term].add(docId)
                else:
                    self.docFreq[term] = {docId}
        self.compute_idf()
        print("preprocessing done")

    def compute_idf(self):
        self.totalDocument
        for term in self.docFreq.keys():
            value = float(self.totalDocument/len(self.docFreq[term]))
            val = math.log(value)
            self.idf[term] = val

    def searchResultBasedOnTf_IdfValues(self, query):
        tfIdfOfDocumentsForSearchedQuery = {}
        docIdsHavingQueryTerms = set()
        queryTerms = word_tokenize(self.docuDataPruning(query).strip())
        for term in queryTerms:
            docIdsHavingQueryTerms = docIdsHavingQueryTerms.union(self.docFreq[term])
        for docId in docIdsHavingQueryTerms:
            tfValueMap = {}
            idfValueMap = {}
            totalTfIdfValueOfQuery = 0
            for term in queryTerms:
                if term in tfValueMap:
                    tfValueMap.update(term,self.tf[term][docId])
                else:
                    if term in self.tf and docId in self.tf[term]:
                        tfValueMap[term] = self.tf[term][docId]
                    else:
                        tfValueMap[term] = 0
                if term in idfValueMap:
                    idfValueMap.update(term,self.idf[term])
                else:
                    if term in self.idf:
                        idfValueMap[term] = self.idf[term]
                    else:
                        idfValueMap[term] = 0
                totalTfIdfValueOfQuery = totalTfIdfValueOfQuery + tfValueMap[term]*idfValueMap[term]

            tfidfValueMap = {}
            tfidfValueMap["tf"] = tfValueMap
            tfidfValueMap["idf"] = idfValueMap
            tfidfValueMap["totalTfIdfValue"] = totalTfIdfValueOfQuery
            tfIdfOfDocumentsForSearchedQuery[docId] = tfidfValueMap
        return tfIdfOfDocumentsForSearchedQuery
        # return sorted(tfIdfOfDocumentsForSearchedQuery.items(), key=lambda k_v: k_v[1]['totalTfIdfValue'], reverse=True)[:10]

    def getDocuText(self, text, searchQuery):
        queryTerms = word_tokenize(str(self.docuDataPruning(searchQuery.strip()).strip()))
        resultData = ""
        self.originalData = word_tokenize(text)
        for word in self.originalData:
            prunedWord = self.docuDataPruning(word).strip()
            if prunedWord in queryTerms:
                resultData = resultData + '<span style="background:yellow;">' + word + '</span> '
            else:
                resultData = resultData + word + ' '
        return resultData

    def callSearch(self, searchQuery):
        self.tfIdfOfDocumentsForSearchedQuery.clear()
        print("in the call search")
        topKDocIds = {}
        searchedResult = {}
        topKDocIds.clear()
        searchedResult.clear()
        synName = set()
        synName.clear()
        
        searchedResult = self.searchResultBasedOnTf_IdfValues(searchQuery)
        print("Total results for query: "+ searchQuery +" is " + str(len(searchedResult)))
        topKDocIds = sorted(searchedResult.items(), key=lambda k_v: k_v[1]['totalTfIdfValue'], reverse=True)[:10]
        resultSet = []
        if len(topKDocIds) > 0:
            for (docId, tfCalculationMap) in topKDocIds:
                if tfCalculationMap["totalTfIdfValue"] > 0:
                    phoneJsonData = {"tfIdfValuesMap":tfCalculationMap,"caption":Markup(str(self.getDocuText(self.original_dataset[docId][2], searchQuery+' '+' '.join(synName)))),"url":Markup(str('<img src="'+self.original_dataset[docId][1]+'" alt="datamining" style="width:20%;height:20%;">'))}
                    resultSet.append(phoneJsonData)
        return resultSet

    def firstfunctionToBeCalled(self):
        self.readCSVFile()
        self.preprocess()