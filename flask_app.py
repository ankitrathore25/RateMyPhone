from flask import Flask, render_template, url_for, request
from preprocessData import mainFile as obj
# from reviewNBClassifier import naiveBayesClassifier as c
from reviewNBClassifier import naiveBayesClassifier as c
from imageClassifier import imageMain as img
import csv

app = Flask(__name__)

mainObj = obj()
classifier = c()
caption = img()

@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('mainpage.html')

@app.route('/search')
def rendeSearchPage():
    return render_template('search.html')

@app.route('/searchQuery', methods=['POST'])
def getSearchQuery():
    global mainObj
    q = request.form['query']
    if 'isSynonymSearch' in request.form:
        isSynonymSearch = True
    else:
        isSynonymSearch = False
    print("searched query: "+q + "isSynonymSearch: "+str(isSynonymSearch))
    resultSet = mainObj.callSearch(q, isSynonymSearch)
    # return render_template('test.html',q=q, result=resultSet)
    return render_template('search.html',q=q, result=resultSet)

@app.route('/classify')
def rendeClassifyPage():
    return render_template('classify.html')

@app.route('/classifyQuery', methods=['POST'])
def classifyText():
    global classifier
    q1 = request.form['query']
    print("text for classification: "+q1)
    resultSet = classifier.classify(q1)
    return render_template('classify.html',q1=q1, classifyResult=resultSet)

@app.route('/image')
def rendeImageCaptioningPage():
    return render_template('imageCaption.html')

@app.route('/imageSearch', methods=['POST'])
def imageSearch():
    global caption
    q1 = request.form['query']
    print("text for classification: "+q1)
    resultSet = caption.callSearch(q1)
    return render_template('imageCaption.html',q1=q1, imageResult=resultSet)

@app.route('/evaluate', methods=['GET'])
def evaluateModel():
    global classifier
    classifier.evaluate_test_data()

if __name__=="__main__":
    #firstfunctionToBeCalled()
    app.run(debug=True, use_reloader=False)
