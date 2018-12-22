#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy as np
import math
from collections import Counter
import os
np.set_printoptions(threshold=np.nan)

class Vectorizer:
	def __init__(self, min_word_length=3, max_df=1.0, min_df=0.0):
		self.min_word_length = min_word_length
		self.max_df=max_df
		self.min_df=min_df
		self.term_df_dict = {}

	def fit(self, raw_documents):
		"""Generates vocabulary for feature extraction. Ignores words shorter than min_word_length and document frequency
		not between max_df and min_df.

		:param raw_documents: list of string for creating vocabulary
		:return: None
		"""
		self.document_count = len(raw_documents)
		self.vocabulary=[]
		self.term_df_dict.clear()
		
		for document in raw_documents:	
			splittedList = document.split(' ')
			
			tempDict={}

			for word in splittedList:
				if(len(word)<self.min_word_length):
					continue
				if(word in self.term_df_dict):
					if not(word in tempDict):
						self.term_df_dict[word] += 1
						tempDict[word]=0
					continue
				self.vocabulary.append(word)
				tempDict[word]=0
				self.term_df_dict[word]=1
		
			tempDict.clear()

		tempp=len(self.vocabulary)
		for i in range(1-tempp,1):
			currentWord=self.vocabulary[-i]
			#print("fitting process: "+str((i+tempp)/float(tempp)) ) 
			if not(self.min_df <= float(self.term_df_dict[self.vocabulary[-i]]) / float(self.document_count) <= self.max_df ):
				del self.vocabulary[-i]
				del self.term_df_dict[currentWord]
				
		pass

	def _transform(self, raw_document, method):
		"""Creates a feature vector for given raw_document according to vocabulary.

		:param raw_document: string
		:param method: one of count, existance, tf-idf
		:return: numpy array as feature vector
		"""


		splittedList=raw_document.split(' ')
		if(method=='existance'):
			returnArray=np.zeros((len(self.vocabulary)),int)
			for word in splittedList:
				if word in self.term_df_dict:
					indexOfWordInVocab = self.vocabulary.index(word)
					returnArray[indexOfWordInVocab]=1
			return returnArray

		if(method=='count'):
			returnArray=np.zeros((len(self.vocabulary)),int)
			for word in splittedList:
				if word in self.term_df_dict:
					indexOfWordInVocab = self.vocabulary.index(word)
					returnArray[indexOfWordInVocab] += 1
			return returnArray

		if(method=='tf-idf'):
			returnArray=np.zeros((len(self.vocabulary)))
			for word in splittedList:
				if word in self.term_df_dict:
					indexOfWordInVocab = self.vocabulary.index(word)
					tf=(float(splittedList.count(word))/float(len(splittedList)))
					returnArray[indexOfWordInVocab]=tf
			for i in range(0, len(returnArray)):
				currentIdf=math.log( float(1+self.document_count) / float(1 + self.term_df_dict[self.vocabulary[i]]) ) + 1
				returnArray[i] *= currentIdf

			sumOfPower2 = 0
			for i in range(0,len(returnArray)):
				sumOfPower2 += returnArray[i]**2
			sumOfPower2 = math.sqrt(sumOfPower2)
			if(sumOfPower2>0.0):
				for i in range(0,len(returnArray)):             
					returnArray[i] /= sumOfPower2
			return returnArray
		
		pass

	def transform(self, raw_documents, method="tf-idf"):
		"""For each document in raw_documents calls _transform and returns array of arrays.

		:param raw_documents: list of string
		:param method: one of count, existance, tf-idf
		:return: numpy array of feature-vectors
		"""


		if(method=="tf-idf"):
			totalNumOfDocs = len(raw_documents)
			featureVectors=np.zeros((totalNumOfDocs,len(self.vocabulary)))
			for i in range(0,totalNumOfDocs):
				featureVectors[i]=Vectorizer._transform(self,raw_documents[i],method)
				#print("transforming process: " +str(float(i)/totalNumOfDocs))
			return featureVectors
				
		else:
			totalNumOfDocs=len(raw_documents)
			featureVectors=np.zeros((totalNumOfDocs,len(self.vocabulary)),int)
			for i in range(0,totalNumOfDocs):
				featureVectors[i]=Vectorizer._transform(self,raw_documents[i],method)
				#print("transforming process: " +str(float(i)/totalNumOfDocs))
			return featureVectors
		
		pass

	def fit_transform(self, raw_documents, method="tf-idf"):
		"""Calls fit and transform methods respectively.

		:param raw_documents: list of string
		:param method: one of count, existance, tf-idf
		:return: numpy array of feature-vectors
		"""


		Vectorizer.fit(self, raw_documents)
		return Vectorizer.transform(self, raw_documents, method)
	
		pass

	def get_feature_names(self):
		"""Returns vocabulary.

		:return: list of string
		"""
		try:
			self.vocabulary
		except AttributeError:
			print "first fit the model"
			return []
		return self.vocabulary

	def get_term_dfs(self):
		"""Returns number of occurances for each term in the vocabulary in sorted.

		:return: array of tuples
		"""
		return sorted(self.term_df_dict.iteritems(), key=lambda (k, v): (v, k), reverse=True)

if __name__=="__main__":
	v = Vectorizer(min_df=0.25, max_df=0.75)
	contents = [
	 "this is the first document",
	 "this document is the second document",
	 "and this is the third one",
	 "is this the first document",
 ]
	
	v.fit(contents)
	print v.get_feature_names()
	existance_vector = v.transform(contents, method="existance")        
	print existance_vector
	count_vector = v.transform(contents, method="count")        
	print count_vector
	tf_idf_vector = v.transform(contents, method="tf-idf")
	print tf_idf_vector
	
	
