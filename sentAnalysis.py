# different approaches to sentiment analysis on movie review dataset
#from kaggle tutorial

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

	model = Word2Vec(sentences, workers=num_workers, \
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

def word2vecAverageForestAnalysis(test,train,num_features):
	#load model
	model = Word2Vec.load("300features_40minwords_10context")
	# clean test and train sets
	clean_train_reviews = cleanReviews(train)
	clean_test_reviews = cleanReviews(test)
	# get average word2vec vectors for cleaned reviews
	avg_train_vecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
	avg_test_vecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
	# train random forest on the training data
	forest = trainForest(avg_train_vecs, train)
	#predict values for test data based on forest
	result = forest.predict(avg_test_vecs)
	#write out results
	predictions = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	predictions.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

def kmeanscluster_word2vec(model):
	word_vectors = model.syn0
	# we're deciding to have about 5 words per cluster
	num_clusters = word_vectors.shape[0]/5
	#initialize a k-means objects and find centroids
	kmeans_clustering = KMeans(n_clusters=num_clusters)
	idx = kmeans_clustering.fit_predict(word_vectors)
	word_centroid_map = dict(zip(model.index2word,idx))
	return word_centroid_map

def create_bag_of_centroids(wordlist, word_centroid_map):
	num_centroids = max(word_centroid_map.values()) +1
	#preallocate for speed
	bag_of_centroids = np.zeros(num_centroids, dtype='float32')
	#loop over words in review, find the cluster that each word belongs to, add to count
	for word in wordlist:
		if word in word_centroid_map.keys(): #some words seem to be missing, maybe tokenizer?
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1
	return bag_of_centroids

def bag_of_centroids_analysis(model, word_centroid_map, train,test):
	word_vectors = model.syn0
	num_clusters = word_vectors.shape[0]/5
	#clean reviews
	clean_test = cleanReviews(test)
	clean_train = cleanReviews(train)

	#create train bag of centroids
	#preallocate for speed
	train_centroids = np.zeros((train['review'].size, num_clusters),dtype='float32')
	counter = 0
	for review in clean_train:
		words = review.split()
		train_centroids[counter] = create_bag_of_centroids(words, word_centroid_map)
	counter += 1

	#create test bag of centroids
	#preallocate
	test_centroids = np.zeros((test['review'].size, num_clusters),dtype='float32')
	counter = 0
	for review in clean_test:
		words = review.split()
		test_centroids[counter] = create_bag_of_centroids(words, word_centroid_map)
		counter +=1

	# fit a random forest

	forest = trainForest(train_centroids, train)

	#predict

	result = forest.predict(test_centroids)

	output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
	output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )





	




	




