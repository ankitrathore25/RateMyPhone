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
# import numpy as np
# import copy
# import pandas as pd
# import pickle
# import re
import math
import csv
# import copy
import operator

class mainFile():
    def __init__(self):
        #  print("commented preprocessing for search engine")
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
        print("reading csv")
        dataset = {}
        self.original_dataset
        #with open('/dataset/amazonPhoneDataset.csv') as csvfile:
        with open('dataset/dataset.csv') as csvfile:
        #with open('output1.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            #[0:'Id', 1:'Product_name', 2:'by_info', 3:'Product_url', 4:'Product_img', 5:'Product_price', 6:'rating',
            # 7;'total_review', 8:'ans_ask', 9:'prod_des', 10:'feature', 11:'cust_review']
            for row in readCSV:
                self.original_dataset[row[0]] = row
                fileData = row[1] + ' ' + row[9] + ' ' + row[10] #this will append multiple text column into one
                if len(fileData.strip()) > 0:
                    dataset.update({row[0]: fileData})
        self.data = dataset
        print("in readCsvFile: "+str(len(self.data)))

    #def testReadCSVFile(self) :
    #    dataset = {}
    #    with open('testFile2.csv') as csvfile:
    #        readCSV = csv.reader(csvfile, delimiter=",")
    #        for row in readCSV:
    #            fileData = row[1]
    #            if len(fileData.strip()) > 0:
    #                dataset.update({row[0]: fileData})
    #    self.data = dataset


    # def convert_lower_case(data):
    #     for (docId, docData) in data.items():
    #         data[docId] = str(np.char.lower(docData))
    #     return data

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
                return document.lower()

    def preprocess(self):
        #self.totalDocument
        #self.data
        for (docId, document) in self.data.items():
            self.allDocIdsList.append(docId)
            self.totalDocument = self.totalDocument + 1
            self.data[docId] = self.docuDataPruning(document)
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

                if term in self.docFreq: #check this part
                    self.docFreq[term].add(docId)
                else:
                    self.docFreq[term] = {docId}
        self.compute_idf()
        print("preprocessing done")
        # compute_tfidf_docu_vector()

    def compute_idf(self):
        self.totalDocument
        for term in self.docFreq.keys():
            value = float(self.totalDocument/len(self.docFreq[term]))
            val = math.log(value)
            self.idf[term] = val

    def compute_tfidf_docu_vector(self):
        #it returns the mapping of each document vector
        #{"doc1":[{"term1":tfidf1},{"term2":tfidf2}],
        # "doc2":[{"term3":tfidf3},{"term4":tfidf4}]}
        # here terms could be same or different, this is used during dot product in cosine similarity
        self.tfIdfDocumentVector
        for docId in self.allDocIdsList:
            tf_idf_vector = {}
            docText = self.data[docId]
            if docText:
                allTermsInDoc = word_tokenize(self.data[docId])
                allUniqueTermInDoc = set(allTermsInDoc)
                for term in allUniqueTermInDoc:
                    tf_idf_vector[term] = (allTermsInDoc.count(term)/self.totalWordsInDoc[docId]) * self.idf[term]
            self.tfIdfDocumentVector[docId] = tf_idf_vector
        return self.tfIdfDocumentVector


    # def computer_tf_idf(query, docId):
    #     tf_idf_score_of_query = 0
    #     for word in query:
    #         tf_value = 0
    #         if word in tf:
    #             if docId in tf[word]:
    #                 tf_value = tf[word][docId]

    #         tf_idf_score_of_query = tf_idf_score_of_query + (tf_value/totalWordsInDoc[docId])*(idf[word])
    #     return tf_idf_score_of_query


    def compute_similarity(self):
        return

    def getQueryVector(self, query):
        queryTokens = word_tokenize(str(self.docuDataPruning(query.strip())))
        uniqueTerms = set(queryTokens)
        queryLength = len(queryTokens)
        queryVector = {}
        for term in uniqueTerms:
            term_count = queryTokens.count(term)
            termFrequency = term_count / queryLength
            if term in self.idf:
                queryIDF = self.idf[term]
            else:
                queryIDF = 0
            tf_idf = termFrequency * queryIDF
            queryVector[term] = tf_idf
        return queryVector

    def getDocumentVector(self, docId):
        docuTokens = word_tokenize(str(self.data[docId]))
        uniqueTerms = set(docuTokens)
        docuLength = len(docuTokens)
        docuVector = {}
        for term in uniqueTerms:
            term_count = docuTokens.count(term)
            termFrequency = term_count / docuLength
            if term in self.idf:
                queryIDF = self.idf[term]
            else:
                queryIDF = 0
            tf_idf = termFrequency * queryIDF
            docuVector[term] = tf_idf
        return docuVector

    def findCosineSimilarity(self, queryVector, queryDocuVector):
        dotProdValue = 0
        for queryTerms in queryVector.keys():
            if queryTerms in queryDocuVector.keys():
                dotProdValue = dotProdValue + queryVector[queryTerms]*queryDocuVector[queryTerms]
        prodOfMod = (norm(list(queryVector.values()))*norm(list(queryDocuVector.values())))
        if prodOfMod == 0:
            return 0
        return dotProdValue/prodOfMod

    def searchResultBasedOnCosineSimilarity(self, query):
        if not query:
            return []
        queryVector = self.getQueryVector(query)
        queryDocuVector = []
        sorted_cosine_similarity = []
        cosine_similarity_query_doc_vector = {}
        for docId in self.allDocIdsList:
            #compute vector for every document
            queryDocuVector = self.getDocumentVector(docId)
            cosine_similarity_query_doc_vector[docId] = (self.findCosineSimilarity(queryVector, queryDocuVector))
        sorted_cosine_similarity =sorted(cosine_similarity_query_doc_vector.items(), key=operator.itemgetter(1),reverse=True)[:10]

        return sorted_cosine_similarity

    def searchResultBasedOnTf_IdfValues(self, query):
        tfIdfOfDocumentsForSearchedQuery = {}
        #global tf
        #global idf
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
            # print("totalTfIdfValueOfQuery"+str(totalTfIdfValueOfQuery))
            tfIdfOfDocumentsForSearchedQuery[docId] = tfidfValueMap
        return tfIdfOfDocumentsForSearchedQuery
        # return sorted(self.tfIdfOfDocumentsForSearchedQuery.items(), key=lambda k_v: k_v[1]['totalTfIdfValue'], reverse=True)[:10]

    def getDocuText(self, text, searchQuery):
        #global data
        #global original_dataset
        queryTerms = word_tokenize(str(self.docuDataPruning(searchQuery.strip()).strip()))
        # docTerms = word_tokenize(str(data[docId].strip()))
        resultData = ""
        # originalData = word_tokenize(original_dataset[docId][1]+" "+original_dataset[docId][9]+" "+original_dataset[docId][10])
        self.originalData = word_tokenize(text)
        for word in self.originalData:
            prunedWord = self.docuDataPruning(word).strip()
            if prunedWord in queryTerms:
                resultData = resultData + '<span style="background:yellow;">' + word + '</span> '
            else:
                resultData = resultData + word + ' '
        return resultData



    def callSearch(self, searchQuery, isSynonymSearch):
        self.tfIdfOfDocumentsForSearchedQuery.clear()
        print("in the call search")
        searQuery = self.docuDataPruning(searchQuery)
        topKDocIds = {}
        searchedResult = {}
        topKDocIds.clear()
        searchedResult.clear()
        synName = set()
        synName.clear()
        if isSynonymSearch:
            #this is implemented only for single word query.
            #can be scaled for long queries too.
            for everyWord in word_tokenize(searQuery.strip()):
                synonyms = wordnet.synsets(everyWord)
                for everySyn in synonyms:
                    synName.add(everySyn.lemmas()[0].name())
                for qry in synName:
                    searchedResult.update(self.searchResultBasedOnTf_IdfValues(qry))
        else:
            searchedResult = self.searchResultBasedOnTf_IdfValues(searchQuery)
        print("Total results for query: "+ searchQuery +" is " + str(len(searchedResult)))
        topKDocIds = sorted(searchedResult.items(), key=lambda k_v: k_v[1]['totalTfIdfValue'], reverse=True)[:10]
        # topKDocIds = searchResultBasedOnCosineSimilarity(searchQuery)
        # print(topKDocIds)
        resultSet = []
        if len(topKDocIds) > 0:
            for (docId, tfCalculationMap) in topKDocIds:
                if tfCalculationMap["totalTfIdfValue"] > 0:
                    # docuText = getDocuText(docId, searchQuery)
                    # print(docuText)
                    phoneJsonData = {"tfIdfValuesMap":tfCalculationMap,"title":Markup(str(self.getDocuText(self.original_dataset[docId][1], searchQuery+' '+' '.join(synName)))),"features":Markup(str(self.getDocuText(self.original_dataset[docId][10], searchQuery+' '+' '.join(synName)))),"desc":Markup(str(self.getDocuText(self.original_dataset[docId][9], searchQuery+' '+' '.join(synName))))}
                    resultSet.append(phoneJsonData)
        # for (docId, similarityValue) in topKDocIds:
        #     if similarityValue > 0:
        #         data = original_dataset[str(docId)]
        #         phoneJsonData = {"similarity":similarityValue,"title":str(data[1]),"features":str(data[10]),"desc":str(data[9])}
        #         resultSet.append(phoneJsonData)
        # print("That's all folk!!!")
        return resultSet

    def firstfunctionToBeCalled(self):
        self.readCSVFile()
        self.preprocess()
        # print("Precomputation Done.... ready to use.......")


    # if __name__=="__main__":
    #     firstfunctionToBeCalled()
    #     callSearch("life learning")
