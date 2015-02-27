# different approaches to sentiment analysis on movie review dataset

import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk.data
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting = 3)
unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter='\t', quoting = 3)
test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting = 3)

def cleanReview(review, remove_stopwords):
	# remove the html
	review_text = BeautifulSoup(review).get_text()
	letters_only = re.sub('[^a-zA-Z]', ' ', review_text) #remove non-letters
	lower_case = letters_only.lower()
	words = lower_case.split()
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
	return words

def joinWordsIntoList(words):
	cleaned_review = ' '.join(words)
	return cleaned_review

def splitReviewIntoSentences(review, tokenizer):
	raw_sentences = tokenizer.tokenize(review.strip()) #split into sentences
	#loop over sentences to clean them
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0: #skip empty sentences
			sentences.append(cleanReview(raw_sentence,remove_stopwords = False))
	return sentences

def cleanReviews(train):
	num_reviews = len(train)
	clean_reviews = []
	for review in train['review']:
		cleaned_words = cleanReview(review, remove_stopwords = True)
		cleaned_review = joinWordsIntoList(cleaned_words)
		clean_reviews.append(cleaned_review)
	return clean_reviews

def getBagOfWords(review_list):
	clean_reviews = cleanReviews(review_list)
	vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
	data_features = vectorizer.fit_transform(clean_reviews)
	data_features = data_features.toarray()
	return data_features

def trainForest(train_data_features, train):
	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(train_data_features, train['sentiment'])
	return forest

def makePredictionsFromForest(forest, test):
	clean_reviews = cleanReviews(test)
	test_features = getBagOfWords(clean_reviews)
	result = forest.predict(test_features)
	return result

def sentences_for_word2vec(train, unlabeled_train):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = []
	for review in train['review']:
		sentences += splitReviewIntoSentences(review, tokenizer)
	for review in unlabeled_train['review']:
		sentences += splitReviewIntoSentences(review, tokenizer)
	return sentences
	
def train_word2vec(sentences):

		# Set values for various parameters
	num_features = 300    # Word vector dimensionality                      
	min_word_count = 40   # Minimum word count                        
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size                                                                                    
	downsampling = 1e-3   # Downsample setting for frequent words

	model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
	model.init_sims(replace=True)

	return model

def averageFeatureVecsOverReview(words, model, num_features):

	#initialize an empy numpy array
	featureVec = np.zeros((num_features,),dtype="float32")
	nwords = 0
	index2words_set = set(model.index2word)

	# loop over every word in the review, add feature vector to total
	for word in words:
		if word in index2words_set:
			featureVec=np.add(featureVec,model[word])
			nwords = nwords+1

	#divide by total
	featureVec = np.divide(featureVec,nwords)
	return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
	#initialize empty array
	featureVecs = np.zeros((len(reviews),num_features),dtype='float32')
	num_reviews = 0
	for review in reviews:
		words = review.split()
		featureVecs[num_reviews] = averageFeatureVecsOverReview(words, model, num_features)
		num_reviews = num_reviews + 1
	return featureVecs

def word2vecAverageForestAnalysis(test,train,model, num_features):
	# clean test and train sets
	clean_train_reviews = cleanReviews(train)
	clean_test_reviews = cleanReviews(test)
	# get average word2vec vectors for cleaned reviews
	avg_train_vecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
	avg_test_vecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
	# train random forest on the training data
	forest = trainForest(avg_train_vecs, train)
	#predict values for test data based on forest
	result = forest.predict(ave_test_vecs)
	#write out results
	predictions = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	predictions.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

def kmeanscluster_word2vec(model):
	word_vectors = model.syn0
	num_clusters = word_vectors.shape[0]/5
	



	




