# USAirlinesSentimentAnalysis

Access the full notebook [here](https://colab.research.google.com/drive/1OGGfLQ9Ur6fyCHWXKFHErDpGFFXi0vXX#scrollTo=5d3f0558).

## Table of Contents:
- [Project Objectives](#project-objective)
- [Project Outcomes](#project-outcomes)
- [Understanding-the-Data](#understanding-the-data)
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

**README in progress**

