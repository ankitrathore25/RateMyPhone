from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def readCSVFile() :
    vid=pd.read_csv("Amazon_Unlocked_Mobile_10k.csv")
    score = []
    X_train, X_test, y_train, y_test = train_test_split(vid['review'], vid['rating'], random_state=1)
    cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    # naive_bayes = MultinomialNB(alpha=0.01)
    # naive_bayes.fit(X_train_cv, y_train)
    # predictions = naive_bayes.predict(X_test_cv)
    # print("Total size: 10000 "+"Train: "+str(X_train_cv.shape[0])+" "+"Test: "+str(X_test_cv.shape[0]))
    # print("Split ratio: 75:25")
    # print('Precision score: ', precision_score(y_test, predictions, average='macro'))
    # print('Recall score: ', recall_score(y_test, predictions, average='macro'))
    # print("Naive Bayes Classifier >> "+"Accuracy: ",accuracy_score(predictions, y_test)*100) 

    # kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    # for kernel in kernels:
    #     SVM = svm.SVC(C=1.0, kernel=kernel, degree=5, gamma='auto')
    #     SVM.fit(X_train_cv,y_train)
    #     predictions_SVM = SVM.predict(X_test_cv)
    #     print("SVM Classifier> kernel: "+kernel+" >> "+"Accuracy: ",accuracy_score(predictions_SVM, y_test)*100)
    
    # ks = [1,3,5,10,12,15,17,20,50,80,100]
    # for k in ks:
    #     neigh = KNeighborsClassifier(n_neighbors=k)
    #     neigh.fit(X_train_cv, y_train)
    #     predictions = neigh.predict(X_test_cv)
    #     print("KNN Classifier >> "+"k: "+str(k)+" Accuracy: "+str(accuracy_score(predictions, y_test)*100))
    #     score.append(accuracy_score(predictions, y_test)*100)

    # cs = [0.1, 1, 5, 10, 15, 20, 50, 100, 1000]
    # for c in cs:
    #     SVM = svm.SVC(C=c, kernel='rbf', gamma='auto')
    #     SVM.fit(X_train_cv,y_train)
    #     predictions_SVM = SVM.predict(X_test_cv)
    #     print("SVM Classifier> kernel: rbf, C: "+ str(c) +" >> "+"Accuracy: ",accuracy_score(predictions_SVM, y_test)*100)
    #     score.append(accuracy_score(predictions_SVM, y_test)*100)

    # degrees = [0, 1, 2, 3, 4, 5, 6]
    # for degree in degrees:
    #     SVM = svm.SVC(C=1.0, kernel='poly', degree=degree, gamma='auto')
    #     SVM.fit(X_train_cv,y_train)
    #     predictions_SVM = SVM.predict(X_test_cv)
    #     print("SVM Classifier> kernel: poly, degree: "+str(degree)+" >> "+"Accuracy: ",accuracy_score(predictions_SVM, y_test)*100)
    #     print()
    
    

    # plotting the graph 
    label = ['Naive Bayes','SVM','KNN']
    score1 = [60.76,59.72,53.60]
    plt.plot(label,score1)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()
    

if __name__=="__main__":
    readCSVFile()
