# RateMyPhone
Classifying the user review to one of the rating class. User will enter the reviews as the plain text in the input box and end result will be the rating based on the classifying model used.

# Summary
- This is my final project for the Data Mining course at UTA. In this project I have to implement 3 phases
- Search : Here user can search for the mobile phones based on the free form search query. User can search for any query, if any similar data is found in the dataset then it will be showed as result.
- Classifier : In this phase user's review will be classified as one of the rating class. Here user give the review of the phone as the free form text and classifier will be able to classify the rating based on the review.
- Image Captioning : This phase is not related to the project but this option will let user to generate the captions for the uploaded images.
- I will use two different datasets. one for search and classifier and another for captioning.
Home Screen will be look like.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/AppHomePage.png)

Search screen will be look like this.
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/searchScreen.png)

# Background Implementation
- Search
For implementing search i've used Tf-IDF method to search for the documents related to the search query entered by the user from website.
I am using flask framework to connect the serverside python code to the frontend part

Calculating the TF-IDF for every term in the query might be slow, so, to reduce the computation time I used precomputation method:
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/tfidf.png)

- In precomputation I'll precompute the term-frequency of every term in every document in the dataset and keep it as a 2D matrix. 
- I'll precompute all the values of the variable needed to find tf-idf of every term

Cosine Similarity
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/cosine_sim.png)
Formula to find cosine similarity:
![](https://github.com/ankitrathore25/RateMyPhone/blob/master/img/similarity.png)
- After finding all the tf-idf values of search query for every document, i'll find the cosine similarity between the query and the documents.
For this i'll find the dot product of search query vector and every document and save it in a list.
Then i'll show the top 10 results having high cosine similarity values.

Right now my search is little bit slow and i am working to optimize the computation of cosine similarity.

Highlighting the search query
-I have done highlighting with simple css property. I added <span style="background:yellow;"></span> around the every term from search query to the the showed text documents. I have used Markup/markupsafe depedency to send HTML code to the flask.

# Demo
- live version of webapp http://dm3.pythonanywhere.com/, https://ratemyphone-api-heroku.herokuapp.com/

# Reference
- https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
- https://github.com/BhaskarTrivedi/QuerySearch_Recommentation_Classification
- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/

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
