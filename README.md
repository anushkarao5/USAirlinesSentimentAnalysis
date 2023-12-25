# USAirlinesSentimentAnalysis

Access the full notebook [here](https://colab.research.google.com/drive/1OGGfLQ9Ur6fyCHWXKFHErDpGFFXi0vXX#scrollTo=5d3f0558).

## Table of Contents:
- [Project Objectives](#project-objective)
- [Project Outcomes](#project-outcomes)
- [Understanding the Data](#understanding-the-data)
- [Text Preprocessing](#text-preprocessing)
- [Text Vectorization](#text-vectorization)
- [Evaluation Metrics](#evaluation-metrics)
- [Modeling with Non NN Models](#modeling-with-non-nn-models)

## Project Objectives: 
The objectives of this project are to:
- Develop a model that accurately classifies tweets regarding 6 U.S. Airlines into positive, negative, and neutral classes
- Discern which text vectorization/ word embedding technique results in the best performance for each classifier

## Project Outcomes 
- Developed a logistic regression classifier using bag of words text vectorization that resulted in an average minority class recall of 73% and overall accuracy of 77%
- Developed a convolutional neural network using pre-trained GloVe word embeddings that resulted in an average minority recall of 68.3%
- Developed an LSTM neural network using pre-trained GloVe word embeddings that resulted in an average minority recall of 67.5%

## Understanding the Data

- This data was taken from Kaggle’s Twitter US Airline Sentiment dataset. 
- The dataset consists of 14,640 Twitter reviews of 6 various airlines. Each of the reviews is labeled as positive, negative, or neutral.

<p align="center">
  <img src="Images/SentDist.png" alt="Image Alt Text" width="500px" height="auto">
</p>

As we can see there are far more instances of negative tweets than of positive and neutral tweets. We expect our classifier to best be able to detect negative tweets because it has the most instances of that class to learn from. 

<p align="center">
  <img src="Images/sentdistperairline.png" alt="Image Alt Text" width="500px" height="auto">
</p>

Among all airlines, the distribution of tweets across the classes is imbalanced. A majority of the reviews are negative. 

## Text Preprocessing 
 What is the purpose of text preprocessing?
- Feeding in cleaner data should yield better results in our models
- We can think of this as "normalizing" our text data
- Note that different models may perform better with/ without certain preprocessing steps
- For example, neural networks often perform better when words are not stemmed, as neural networks can learn complex patterns directly from raw text
- However, since we will begin with non-neural network machine learning models, we will thoroughly preprocess the text before inputting it into our model

<p align="center">
  <img src="Images/pre_function.png" alt="Image Alt Text" width="800px" height="auto">
</p>

Our preprocessor function cleans HTML tags, removes handles and URLs, strips punctuation, removes stop words, and stems words. 

<p align="center">
  <img src="Images/pre_exm_tweet.png" alt="Image Alt Text" width="800px" height="auto">
</p>

An example of how our preprocessor function works when applied to a tweet. We can think of this step as making the text cleaner before vectorization.

## Text Vectorization 

Many models cannot handle text data as it is. Therefore we must convert the words into vectors of numbers that the model can interpret. There are several ways we can do this. For non NN models, we will use Bag of Words and TFIDF Vectorization. Later, for the NNs, we will discuss alternative word vectorization options.

**Bag of Words Vectorization**
- Each unique word is represented as a feature, and each tweet is represented as a row
- We put a 1 if the word is present in the tweet, and a 0 if the word is not present (one-hot encoding for text)
- Let's take a simple corpus with three sentences that we would like to vectorize:

<p align="center">
  <img src="Images/mini_corpus.png" alt="Image Alt Text" width="500px" height="auto">
</p>

We can use the count vectorizer function from sklearn to produce a one-hot encoded data frame with the rows as sentences in the corpus and the columns as the unique words in the corpus.

<p align="center">
  <img src="Images/BOW.png" alt="Image Alt Text" width="500px" height="auto">
</p>

Now each tweet is a vector of 0s and 1s. Now we can feed these vectors into our models. 

**TFIDF Vectorization**

- Term frequency-inverse document frequency
- The basic idea: the more times a given term appears in a document (a particular sentence), the more important the word is to understand the document
- At the same time, terms that appear in almost every document are likely not important in understanding a specific document
- TF-IDF factors in both of these concerns

Since this involves a slightly more complex calculation, I will not be going into detail about it here. However, if you are interested in learning more about TFIDF, click [here](https://colab.research.google.com/drive/1OGGfLQ9Ur6fyCHWXKFHErDpGFFXi0vXX#scrollTo=4c1665d4). 

<p align="center">
  <img src="Images/TFIDF.png" alt="Image Alt Text" width="500px" height="auto">
</p>

This is how we would represent each sentence in the corpus using TFIDF scores. As we can see, words that are important to a specific document have a higher score: example "cats" for document 1, "dogs" for document 2, and "coolest" for document 3. Words that appear in all documents like "are" have the lowest scores.


**README in progress**

