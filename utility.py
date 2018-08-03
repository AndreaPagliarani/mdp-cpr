# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:30:56 2018

@author: andreap
"""
# pre-processing utilities
import re, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer

class Preprocessing:
    """
    This class contains data pre-processing utilities.
    """
    def __init__(self, stemming=False, lemmatization=False, stemmer='snowball'):
        self.stemming=stemming
        self.lemmatization=lemmatization
        self.stemmer=PorterStemmer() if stemmer=='porter' else LancasterStemmer() if stemmer=='lancaster' else SnowballStemmer('english')

    def preprocess_data(self, text, lowercase=True, punctuation_removal=True, number_removal=True, stopwords_removal=True, min_word_length=1):
        text = self.denoise_text(text)
        text = self.replace_contractions(text)
        words = nltk.word_tokenize(text)
        p = inflect.engine()
        if self.lemmatization:
            lemmatizer = WordNetLemmatizer()
        # normalize words
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') # remove non ASCII
            if lowercase is True:
                new_word = new_word.lower()
            if new_word.replace('.','').replace(',','').isdigit():
                if number_removal is True:
                    continue
                else:
                    new_word = p.number_to_words(new_word)
            if punctuation_removal is True:
                new_word = re.sub(r'[^\w\s]', ' ', new_word)
            if new_word.strip() == '':
                continue
            if stopwords_removal is True and new_word in stopwords.words('english'):
                continue
            if len(new_word) < min_word_length: 
                continue
            if self.stemming:
                new_word = self.stemmer.stem(new_word)
            elif self.lemmatization:
                tag = nltk.tag.pos_tag([new_word])[0][1]
                tag = self.get_wordnet_pos(tag)
                if tag != '':
                    new_word = lemmatizer.lemmatize(new_word, pos=tag)
            new_words.append(new_word)
        return new_words
        
    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    
    def remove_between_square_brackets(self, text):
        return re.sub('\[[^]]*\]', ' ', text)
    
    def remove_linebreaks(self, text):
        return re.sub('[\\\\]+.', ' ', text)
    
    def denoise_text(self, text):
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_linebreaks(text)
        return text
    
    def replace_contractions(self, text):
        """Replace contractions in string of text"""
        return contractions.fix(text)
    
    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    def replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    def remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
    
    def get_wordnet_pos(self, tag): 
        if tag.startswith('J'): 
            return wordnet.ADJ 
        elif tag.startswith('V'): 
            return wordnet.VERB 
        elif tag.startswith('N'): 
            return wordnet.NOUN 
        elif tag.startswith('R'): 
            return wordnet.ADV 
        else: 
            return ''
    
    def stem_words(self, words):
        """Stem words in list of tokenized words"""
        stems = []
        for word in words:
            stem = self.stemmer.stem(word)
            stems.append(stem)
        return stems
    
    def lemmatize_words(self, words):
        """Lemmatize words in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            tag = nltk.tag.pos_tag([word])[0][1]
            tag = self.get_wordnet_pos(tag)
            if tag != '':
                lemma = lemmatizer.lemmatize(word, pos=tag)
                lemmas.append(lemma)
        return lemmas
    
    def remove_shortwords(self, words, min_length):
        new_words = []
        for word in words:
            if len(word) >= min_length:
                new_words.append(word)
        return new_words
    
    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words
    
    def stem_and_lemmatize(self, words):
        stems = self.stem_words(words)
        lemmas = self.lemmatize_words(words)
        return stems, lemmas