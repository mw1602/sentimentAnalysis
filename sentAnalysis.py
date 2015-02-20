# sentiment analysis on movie review dataset

import pandas as pd 
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting = 3)
unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter='\t', quoting = 3)
test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting = 3)

def cleanReview(review):
	# remove the html
	review = BeautifulSoup(review)
	review_text = review.get_text()
	letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
	lower_case = letters_only.lower()
	words = lower_case.split()
	stops = set(stopwords.words("english"))
	words = [w for w in words if not w in stops]
	cleaned_review = ' '.join(words)
	return cleaned_review

def cleanReviews(train):
	num_reviews = len(train)
	clean_reviews = []
	for review in train['review']:
		cleaned_review = cleanReview(review)
		clean_reviews.append(cleaned_review)
	return clean_reviews

def getBagOfWords(review_list):
	clean_reviews = cleanReviews(review_list)
	vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
	data_features = vectorizer.fit_transform(clean_reviews)
	data_features = data_features.toarray()
	return data_features

def trainForest(train_data_features, train):
	forest = RandomForestClassifier(n_estimaters = 100)
	forest = forest.fit(train_data_features, train['sentiment'])
	return forest

def makePredictions(forest, test):
	clean_reviews = cleanReviews(test)
	test_features = getBagOfWords(clean_reviews)
	result = forest.predict(test_features)
	return result

