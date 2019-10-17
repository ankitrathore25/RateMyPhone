# RateMyPhone
Classifying the user review to one of the rating class. User will enter the reviews as the plain text in the input box and end result will be the rating based on the classifying model used.

# Summary
- This is my semester project for the Data Mining course at UTA. In this project I have to implement 3 phases
- 1) Search : Here user can search for the mobile phones based on the free form search query. User can search for any query, if any similar data is found in the dataset then it will be showed as result.
- 2) Classifier : In this phase user's review will be classified as one of the rating class. Here user give the review of the phone as the free form text and classifier will be able to classify the rating based on the review.
- 3) Image Captioning : This phase is not related to the project but this option will let user to generate the captions for the uploaded images.
- I have used different datasets for every phase. The reason for choosing different dataset is that I couldn't find a single dataset which have text column (which can be used for search) and images(used for image captioning) associated with the data.
Home Screen will be look like.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/AppHomePage.png)

Search screen will be look like this.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/searchScreen.png)

# Background Knowledge
# Search
I have implemented Tf-IDF(Term Frequency–Inverse Document Frequency) based ranking system for search engine. In information retrieval, TF-IDF of a term shows how important a word is in a pile of documents.
Main Idea: For every key word from search query we find the whether that term exist in the document or not if it exist in document then what is its impact relative to the frequency of term in that document and in other document.

tf = (frequnecy of a word in the document) / (total no. of words in that document)
idf = (total number of documents) / (number of documents in which term appears)
We used normalization for calculating idf because value of total number of documents could be very large which will make the calculation a little bit tedious, so we use logarithm of the idf value.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/tfidf.png)

# Stopword Removal
There are some words which are too common in english language like "the","is","a". These words are called stopwords which contain least meaning. An example from Wall Street Journal.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/wordVsFrequencyTable.png)
Our first step is to remove these unneccessary words from the corpus so that our calculation becomes a little bit easier. Common stopwords in english are:
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/stopwordInEnglish.png)
We used python nltk stopword corpus for removing stopwords.

# Stemming
Another step in pruning data is to reduce the words to their stems word called Stemming. Take for example words fishing, fished, and fisher are related to fish(stem word) but without reducing to the stem word these are considered different. So, next step is to reduce every word in the corpus to their stem word. In python, we have many different stemmers like Potter Stemmer, Snowbll Stemmer, Lancaster Stemmer. Difference between these stemmers are very well explained in the answer https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg.

# Experiemental Result
- I have tested different stemmer for their effect on the search result in my case. Below table explains it:
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/stemmerComparison.png)

We can see that how changing stemmer changes the search result so it is important to choose stemmer wisely.

# Lemmatization
Lemmatization is also similar to stemming but with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word.
I have used nltk wordnet lemmatizer in my code. For processing data sequence of these techniques are important. I have first removed stopwords then applied lemmatization and then stemming. If I apply lemmatization before the stemming then result will be different.

# Challenge Faced
- First challenge is to calculate the tf-idf value of every term for every documents in the corpus. Now, you must be able to visualize how time consuming it would be if I start calculating tf of every term in every document and it will be very costly if I do this tf-idf calcuation for every searched query.
> To solve this problem I have used pre-computation method. The idea is I'll process the data at the time of server start(only once) and keep a matrix of term vs document which will hold the count of every term in the corpus with every document, I will also keep a documentFrequency map which will hold the value of number of documents in which that term is present. I've used some other variables too just to avoid unneccessary computation again and again. So, using all these pre-computed values we can find the value of tf-idf of terms in O(1) time. Only trade off of this method is space complexity increases. We need to keep the values during the runtime but it will allow fast computation of tf-idf values for every docuement.
We will then show the top records based on tf-idf values of the documents.

# Cosine Similarity
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/cosine_sim.png)
Formula to find cosine similarity:
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/similarity.png)
- After finding all the tf-idf values of search query for every document, i'll find the cosine similarity between the query and the documents.
For this i'll find the dot product of search query vector and every document and save it in a list.
Then i'll show the top 10 results having high cosine similarity values.

Right now my search is little bit slow and i am working to optimize the computation of cosine similarity.

# Challenge Faced
- Second challenge I faced is calculating cosine similarity of the query and documents. It takes huge time in calculating dot product of query vector with every document. I will be handling this problem with calculating dot product with top K documents with the query. These top K will be selected on the basis of idf value for the query terms.

# Highlighting the search query
- I have done highlighting with simple css property. I added <span style="background:yellow;"></span> around the every term from search query to the the showed text documents. I have used Markup/markupsafe depedency to send HTML code to the flask.

# Synonyms and Phrase support:
I have implemented synonyms support, for now this is available for only single word but this can be scaled to the multiple words query. Reason for keeping it to single word is because I am searching for every synonyms of the word so for example my search query is "best phones", we have more than 10 unique synonyms in nltk wordnet and 4 synonyms for the phone so for all the synonyms possible queries I need to process will be 10*4, which will take some quite time.
I know this is not the good approach but I'll keep it for future optimization.

# More Challenges Faced
- Third challenge I faced was hosting my app to online freely available webserver (pythonanywhere or heroku). I know it is very easy to host online but I struggled with setting values in the global variables I defined. On these web hosting servers we do not run app using app.run() and flask is not thread safe so it creates different set of global variables and these global variables are inaccessible in the other functions. After restructuring the code and keeping all the code in a class which can be accessed using object solved this problem.

# Demo
- live version of webapp http://dm3.pythonanywhere.com/, https://ratemyphone-api-heroku.herokuapp.com/

# Reference
- https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
- https://github.com/BhaskarTrivedi/QuerySearch_Recommentation_Classification
- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
- https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg
- https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python

# Dataset
- https://www.kaggle.com/ak47bluestack/amazonphonedataset

# point to note
- install all the dependencies of nltk before deploying on server

# Technology
- Python
- Flask
- HTML
- CSS
- Git
