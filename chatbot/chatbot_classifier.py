#https://chatbotslife.com/text-classification-using-algorithms-e4d50dcba45

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords


class ChatbotClassifier():

	def __init__(self):
		self.stemmer = LancasterStemmer()
		self.training_data = self.load_training_data()
		self.current_sentence = ""
		self.corpus_words = {}
		self.class_words = {}

		self.tokenise_sentences()


	def load_training_data(self):
		data = []
		import os
		# read all data files and add them to the list
		for data_file in os.listdir("training_data"):
			with open("training_data/" + data_file, "r") as file:
				for line in file:
					data.append({"class":data_file.replace(".txt",""), "sentence":line.strip("\n")})

		return data


	def tokenise_sentences(self):
		# turn a list into a set (of unique items) and then a list again (this removes duplicates)
		classes = list(set([a['class'] for a in self.training_data]))
		for c in classes:
		    # prepare a list of words within each class
		    self.class_words[c] = []

		# loop through each sentence in our training data
		for data in self.training_data:
		    # tokenize each sentence into words
		    for word in nltk.word_tokenize(data['sentence']):
		        
		        # ignore a some things
		        if word not in stopwords.words("english"):
		            # stem and lowercase each word
		            stemmed_word = self.stemmer.stem(word.lower())
		            # have we not seen this word already?
		            if stemmed_word not in self.corpus_words:
		                self.corpus_words[stemmed_word] = 1
		            else:
		                self.corpus_words[stemmed_word] += 1

		            # add the word to our words in class list
		            self.class_words[data['class']].extend([stemmed_word])



	# calculate a score for a given class taking into account word commonality
	def calculate_class_score(self, class_name, show_details=True):
	    score = 0
	    # tokenize each word in our new sentence
	    for word in nltk.word_tokenize(self.current_sentence):
	        # check to see if the stem of the word is in any of our classes
	        if self.stemmer.stem(word.lower()) in self.class_words[class_name]:
	            # treat each word with relative weight
	            score += (1 / self.corpus_words[self.stemmer.stem(word.lower())])

	    return score


	def classify(self):
	    high_class = None
	    high_score = 0
	    # loop through our classes
	    for c in self.class_words.keys():
	        # calculate score of sentence for each class
	        score = self.calculate_class_score(c, show_details=False)
	        # keep track of highest score
	        if score > high_score:
	            high_class = c
	            high_score = score

	    return high_class, high_score


	def set_sentence(self, sent):
		self.current_sentence = sent

