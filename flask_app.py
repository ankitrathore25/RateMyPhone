from flask import Flask, render_template, url_for, request
from preprocessData import firstfunctionToBeCalled, callSearch
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def getSearchQuery():
    q = request.form['query']
    resultSet = callSearch(q)
    return render_template('test.html',q=q, result=resultSet)

if __name__=="__main__":
    firstfunctionToBeCalled()
    app.run(debug=True)
