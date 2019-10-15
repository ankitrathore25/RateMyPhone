from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
from markupsafe import Markup

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

def readCSVFile() :
    print("reading csv")
    dataset = {}
    global original_dataset
    with open('amazonPhoneDataset.csv') as csvfile:
    # with open('testData.csv') as csvfile:
    # with open('output1.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        #[0:'Id', 1:'Product_name', 2:'by_info', 3:'Product_url', 4:'Product_img', 5:'Product_price', 6:'rating', 
        # 7;'total_review', 8:'ans_ask', 9:'prod_des', 10:'feature', 11:'cust_review']
        for row in readCSV:
            original_dataset[row[0]] = row
            fileData = row[1] + ' ' + row[9] + ' ' + row[10] #this will append multiple text column into one
            if len(fileData.strip()) > 0:
                dataset.update({row[0]: fileData})
    global data 
    data = dataset
    print(len(data))

def testReadCSVFile() :
    dataset = {}
    with open('testFile2.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for row in readCSV:
            fileData = row[1]
            if len(fileData.strip()) > 0:
                dataset.update({row[0]: fileData})
    global data 
    data = dataset


# def convert_lower_case(data):
#     for (docId, docData) in data.items():
#         data[docId] = str(np.char.lower(docData))
#     return data

def remove_stop_words(document):
    if not document:
        return ""
    stop_words = stopwords.words('english')
    words = word_tokenize(str(document))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w.strip()
    return new_text.strip()


def stemming(document):
    if not document:
        return ""
    stemmer= PorterStemmer()
    tokens = word_tokenize(str(document))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def lemmatize(document):
    if not document:
        return ""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(document))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w,pos="v")
    return new_text

def docuDataPruning(document):
    if not document:
        return ""
    else:
        document = remove_stop_words(document)
        document = lemmatize(document)
        document = stemming(document)
        if not document:
            return ""
        else:
            return document.lower()

def preprocess(tf):
    global totalDocument
    global data
    for (docId, document) in data.items():
        allDocIdsList.append(docId)
        totalDocument = totalDocument + 1
        data[docId] = docuDataPruning(document)
        terms = word_tokenize(data[docId])
        totalWordsInDoc[docId] = len(terms)
        for term in terms:
            allTermsSet.add(term)
            if term in tf:
                row = tf[term]
                if docId in row:
                    row[docId] = (row[docId] + 1)
                else:
                    row[docId] = 1
            else:
                tf[term] = {str(docId):1}

            if term in docFreq: #check this part
                docFreq[term].add(docId)
            else:
                docFreq[term] = {docId}
    compute_idf()
    # compute_tfidf_docu_vector()

def compute_idf():
    global totalDocument
    for term in docFreq.keys():
        value = float(totalDocument/len(docFreq[term]))
        val = math.log(value)
        idf[term] = val

def compute_tfidf_docu_vector():
    #it returns the mapping of each document vector
    #{"doc1":[{"term1":tfidf1},{"term2":tfidf2}],
    # "doc2":[{"term3":tfidf3},{"term4":tfidf4}]}
    # here terms could be same or different, this is used during dot product in cosine similarity
    global tfIdfDocumentVector
    for docId in allDocIdsList:
        tf_idf_vector = {}
        docText = data[docId]
        if docText:
            allTermsInDoc = word_tokenize(data[docId])
            allUniqueTermInDoc = set(allTermsInDoc)
            for term in allUniqueTermInDoc:
                tf_idf_vector[term] = (allTermsInDoc.count(term)/totalWordsInDoc[docId]) * idf[term]
        tfIdfDocumentVector[docId] = tf_idf_vector
    return tfIdfDocumentVector
        

# def computer_tf_idf(query, docId):
#     tf_idf_score_of_query = 0
#     for word in query:
#         tf_value = 0
#         if word in tf:
#             if docId in tf[word]:
#                 tf_value = tf[word][docId]
        
#         tf_idf_score_of_query = tf_idf_score_of_query + (tf_value/totalWordsInDoc[docId])*(idf[word])
#     return tf_idf_score_of_query


def compute_similarity():
    return

def getQueryVector(query):
    queryTokens = word_tokenize(str(docuDataPruning(query.strip())))
    uniqueTerms = set(queryTokens)
    queryLength = len(queryTokens)
    queryVector = {}
    for term in uniqueTerms:
        term_count = queryTokens.count(term)
        termFrequency = term_count / queryLength
        if term in idf:
            queryIDF = idf[term]
        else:
            queryIDF = 0
        tf_idf = termFrequency * queryIDF
        queryVector[term] = tf_idf
    return queryVector

def getDocumentVector(docId):
    docuTokens = word_tokenize(str(data[docId]))
    uniqueTerms = set(docuTokens)
    docuLength = len(docuTokens)
    docuVector = {}
    for term in uniqueTerms:
        term_count = docuTokens.count(term)
        termFrequency = term_count / docuLength
        if term in idf:
            queryIDF = idf[term]
        else:
            queryIDF = 0
        tf_idf = termFrequency * queryIDF
        docuVector[term] = tf_idf
    return docuVector

def findCosineSimilarity(queryVector, queryDocuVector):
    dotProdValue = 0
    for queryTerms in queryVector.keys():
        if queryTerms in queryDocuVector.keys():
            dotProdValue = dotProdValue + queryVector[queryTerms]*queryDocuVector[queryTerms]
    prodOfMod = (norm(list(queryVector.values()))*norm(list(queryDocuVector.values())))
    if prodOfMod == 0:
        return 0
    return dotProdValue/prodOfMod

def searchResultBasedOnCosineSimilarity(query):
    if not query:
        return []
    queryVector = getQueryVector(query)
    queryDocuVector = []
    sorted_cosine_similarity = []
    cosine_similarity_query_doc_vector = {}
    for docId in allDocIdsList:
        #compute vector for every document
        queryDocuVector = getDocumentVector(docId)
        cosine_similarity_query_doc_vector[docId] = (findCosineSimilarity(queryVector, queryDocuVector))
    sorted_cosine_similarity =sorted(cosine_similarity_query_doc_vector.items(), key=operator.itemgetter(1),reverse=True)[:10]

    return sorted_cosine_similarity

def searchResultBasedOnTf_IdfValues(query):
    global tfIdfOfDocumentsForSearchedQuery
    global tf
    global idf
    docIdsHavingQueryTerms = set()
    queryTerms = word_tokenize(docuDataPruning(query).strip())
    for term in queryTerms:
        docIdsHavingQueryTerms = docIdsHavingQueryTerms.union(docFreq[term])
    for docId in docIdsHavingQueryTerms:
        tfValueMap = {}
        idfValueMap = {}
        totalTfIdfValueOfQuery = 0
        for term in queryTerms:
            if term in tfValueMap:
                tfValueMap.update(term,tf[term][docId])
            else:
                if term in tf and docId in tf[term]:
                    tfValueMap[term] = tf[term][docId]
                else:
                    tfValueMap[term] = 0
            if term in idfValueMap:
                idfValueMap.update(term,idf[term])
            else:
                if term in idf:
                    idfValueMap[term] = idf[term]
                else:
                    idfValueMap[term] = 0
            totalTfIdfValueOfQuery = totalTfIdfValueOfQuery + tfValueMap[term]*idfValueMap[term]
            
        tfidfValueMap = {}
        tfidfValueMap["tf"] = tfValueMap
        tfidfValueMap["idf"] = idfValueMap
        tfidfValueMap["totalTfIdfValue"] = totalTfIdfValueOfQuery
        # print("totalTfIdfValueOfQuery"+str(totalTfIdfValueOfQuery))
        tfIdfOfDocumentsForSearchedQuery[docId] = tfidfValueMap
    return sorted(tfIdfOfDocumentsForSearchedQuery.items(), key=lambda k_v: k_v[1]['totalTfIdfValue'], reverse=True)[:10]

def getDocuText(text, searchQuery):
    global data
    global original_dataset
    queryTerms = word_tokenize(str(docuDataPruning(searchQuery.strip()).strip()))
    # docTerms = word_tokenize(str(data[docId].strip()))
    resultData = ""
    # originalData = word_tokenize(original_dataset[docId][1]+" "+original_dataset[docId][9]+" "+original_dataset[docId][10])
    originalData = word_tokenize(text)
    for word in originalData:
        prunedWord = docuDataPruning(word).strip()
        if prunedWord in queryTerms:
            resultData = resultData + '<span style="background:yellow;">' + word + '</span> '
        else:
            resultData = resultData + word + ' '
    return resultData



def callSearch(searchQuery):
    global tfIdfOfDocumentsForSearchedQuery
    tfIdfOfDocumentsForSearchedQuery.clear()
    print("in the call search")
    topKDocIds = searchResultBasedOnTf_IdfValues(searchQuery)
    # topKDocIds = searchResultBasedOnCosineSimilarity(searchQuery)
    # print(topKDocIds)
    resultSet = []
    if len(topKDocIds) > 0:
        for (docId, tfCalculationMap) in topKDocIds:
            if tfCalculationMap["totalTfIdfValue"] > 0:
                # docuText = getDocuText(docId, searchQuery)
                # print(docuText)
                phoneJsonData = {"tfIdfValuesMap":tfCalculationMap,"title":Markup(str(getDocuText(original_dataset[docId][1], searchQuery))),"features":Markup(str(getDocuText(original_dataset[docId][10], searchQuery))),"desc":Markup(str(getDocuText(original_dataset[docId][9], searchQuery)))}
                resultSet.append(phoneJsonData)
    # for (docId, similarityValue) in topKDocIds:
    #     if similarityValue > 0:
    #         data = original_dataset[str(docId)]
    #         phoneJsonData = {"similarity":similarityValue,"title":str(data[1]),"features":str(data[10]),"desc":str(data[9])}
    #         resultSet.append(phoneJsonData)
    # print("That's all folk!!!")
    return resultSet

def firstfunctionToBeCalled():
    readCSVFile()
    preprocess(tf)
    # print("Precomputation Done.... ready to use.......")
    

 
# if __name__=="__main__":
#     firstfunctionToBeCalled()
#     callSearch("life learning")