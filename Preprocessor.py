#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import os
from nltk.corpus import stopwords
import codecs
import errno

import string

class Preprocessor:
	def __init__(self, dataset_directory="Dataset", processed_dataset_directory= "ProcessedDataset"):
		self.dataset_directory = dataset_directory
		self.processed_dataset_directory=processed_dataset_directory
		nltk.download("stopwords")
		nltk.download("punkt")
		self.stop_words = set(stopwords.words('english'))

	def _remove_puncs_numbers_stop_words(self, tokens):
		"""Remove punctuations in the words, words including numbers and words in the stop_words list.

		:param tokens: list of string
		:return: list of string with cleaned version
		"""

		
		for i in range(-len(tokens)+1,1):
			if(tokens[-i].isalpha()):
				if(tokens[-i] in self.stop_words):
					del tokens[-i]
			else:
				if(tokens[-i].isalnum()):
					del tokens[-i]
				else:
					for characterIndex in range(-len(tokens[-i])+1,1):
						if(tokens[-i][-characterIndex] in string.punctuation):
							tokens[-i] = tokens[-i][:-characterIndex] + tokens[-i][-characterIndex+1:]
					
					if(tokens[-i] in self.stop_words or (not tokens[-i].isalpha() and tokens[-i].isalnum() ) or len(tokens[-i])==0 ):
						del tokens[-i]

		return tokens
	
		pass

	def _tokenize(self, sentence):
		"""Tokenizes given string.

		:param sentence: string to tokenize
		:return: list of string with tokens
		"""

		
		return nltk.word_tokenize(sentence.lower())
	
		pass

	def _stem(self, tokens):
		"""Stems the tokens with nltk SnowballStemmer

		:param tokens: list of string
		:return: list of string with words stems
		"""

		
		resultList=[]
		stemmer= nltk.SnowballStemmer("english")
		for token in tokens:
			token=stemmer.stem(token)
			if(token!=""):
				resultList.append(token)
		return resultList
	
		pass

	def preprocess_document(self, document):
		"""Calls methods _tokenize, _remove_puncs_numbers_stop_words and _stem respectively.

		:param document: string to preprocess
		:return: string with processed version
		"""
		

		
		tokenized = Preprocessor._tokenize(self, document)
		removedPuncsStops = Preprocessor._remove_puncs_numbers_stop_words(self,tokenized)
		removedStems = Preprocessor._stem(self,removedPuncsStops)
		return ' '.join(removedStems)
	
		pass


	def preprocess(self):
		"""Walks through the given directory and calls preprocess_document method. The output is
		persisted into processed_dataset_directory by keeping directory structure.

		:return: None
		"""
		for root, dirs, files in os.walk(self.dataset_directory):
			if os.path.basename(root) != self.dataset_directory:
				print "Processing", root, "directory."
				dest_dir = self.processed_dataset_directory+"/"+root.lstrip(self.dataset_directory+"/")
				if not os.path.exists(dest_dir):
					try:
						os.makedirs(dest_dir)
					except OSError as exc:
						if exc.errno != errno.EEXIST:
							raise
				for file in files:
					file_path = root + "/" + file
					with codecs.open(file_path, "r", "ISO-8859-1") as f:
						data = f.read().replace("\n", " ")
					processed_data = self.preprocess_document(data)
					output_file_path = dest_dir + "/" + file
					with codecs.open(output_file_path, "w", "ISO-8859-1") as o:
						o.write(processed_data)

if __name__=="__main__":
	text =  """ Greetings,
				shall I sit or stand?
				- Tell us.
				- Tell us.
				I'll tell. We bought the goods
				from Black Faik.
				We reloaded the truck
				in Karabuk.
				I was driving the truck
				till Adana.
				- What are you talking about?
				- And you?!
				You've abducted me,
				you'll do the talking.
				I'm confused anyway.
				- Aggressive.
				- Aggressive.
				Yeah, aggressive.
				Is that it?"""

