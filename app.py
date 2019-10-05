from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def getSearchQuery():
    q = request.form['query']
    return render_template('test.html',q=q)

if __name__=="__main__":
    app.run(debug=True)