import pandas as pd
import numpy as np
from CustomException import CustomException
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter, defaultdict
from nltk.util import ngrams
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

import warnings

warnings.filterwarnings("ignore")

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

# Add new stop words
"""
    Create a new list of stop words or append the standard NLTK stop words 
    :listOfStopWords: Python list of words of string type
    :is_new: True for an exclusive list of stopwords/ False to append to NLTK stopword list
    - return: stopword set of words(can be used directly anywhere)
"""
def add_stopwords(listOfStopWords,
                  is_new = False):
    if is_new:
        return set(listOfStopWords)
    else:
        if listOfStopWords is None:
            raise CustomException('Please provide a list of stopwords')
        else:
            return set(list(set(stopwords.words('english')))+listOfStopWords)

# Reduce repeated alphabets to singletons
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# Text Pre processing Cleaner function
"""
    Clean the text based on your requirements 
    :data: Pandas dataframe containing the text
    :column_name: Name of column containing the text
    :listOfStopWords: Python list of words of string type
    :remove_digits: True/False to remove digits(numbers) from text
    :remove_stopwords: True/False to remove stopwords
    :append_stopwords: True/False append stop words to the current NLTK set
    :do_lemmatization: True/False perform lemmatization or not
    - return: Pandas dataframe with a new column with _processed attached to column name provided
"""
def text_cleaner(data,
                column_name,
                listOfStopWords = [],
                remove_digits = True,
                remove_stopwords = True,
                append_stopwords = False,
                do_lemmatization = True):
    if data is None or column_name is None or listOfStopWords is None or remove_stopwords is None or append_stopwords is None or do_lemmatization is None:
        raise CustomException("Please check the arguments, one or more are assigned None value")
    
    new_column_name = column_name+'_processed'
    # htmlSyntax = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    data['temporary'] = data[column_name].apply(lambda x: x.lower())
    data['temporary'] = data['temporary'].apply(lambda x: BeautifulSoup(x, "html.parser").text)
    data['temporary'] = data['temporary'].apply(lambda x: ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in x.split(" ")]))
    data['temporary'] = data['temporary'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\(|\)|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', str(x)))

    if remove_digits:
        data['temporary'] = data['temporary'].apply(lambda x: re.sub('\d+',' ', str(x)))
    data['tokens'] = data['temporary'].apply(lambda x: word_tokenize(x))


    if do_lemmatization:
        lemmatizer = WordNetLemmatizer()
        if remove_stopwords:
            stopwordsList = None
            if (len(listOfStopWords) == 0 or listOfStopWords is None) and append_stopwords == False:
                stopwordsList = set(stopwords.words('english'))
            elif (len(listOfStopWords) == 0 or listOfStopWords is None) and append_stopwords == True:
                raise CustomException('Please provide a list of stop words or set append_stopwords == False')
            elif (len(listOfStopWords) != 0) and append_stopwords == True:
                stopwordsList = add_stopwords(listOfStopWords, is_new = False)
            elif (len(listOfStopWords) != 0) and append_stopwords == False:
                stopwordsList = set(listOfStopWords)
            else:
                raise CustomException('Case is not added into the repository yet')

            data[new_column_name] = data['tokens'].apply(lambda x: ' '.join(reduce_lengthening(lemmatizer.lemmatize(eachWord)) for eachWord in x if eachWord not in stopwordsList))
        else:

            if len(listOfStopWords) > 0 or append_stopwords:
                print(len(listOfStopWords), append_stopwords)
                raise CustomException("Please set the remove_stopwords flag to True or set append_stopwords = False and listOfStopWords = []")
            data[new_column_name] = data['tokens'].apply(lambda x: ' '.join(reduce_lengthening(lemmatizer.lemmatize(eachWord)) for eachWord in x))
    else:       
        if remove_stopwords:
            stopwordsList = None
            if (len(listOfStopWords) == 0 or listOfStopWords is None) and append_stopwords == False:
                stopwordsList = set(stopwords.words('english'))
            elif (len(listOfStopWords) == 0 or listOfStopWords is None) and append_stopwords == True:
                raise CustomException('Please provide a list of stop words or set append_stopwords = False')

            elif (len(listOfStopWords) != 0) and append_stopwords == True:
                stopwordsList = add_stopwords(listOfStopWords, is_new = False)
            elif (len(listOfStopWords) != 0) and append_stopwords == False:
                stopwordsList = set(listOfStopWords)
            else:
                raise CustomException('Case is not added into the repository yet')
            data[new_column_name] = data['tokens'].apply(lambda x: ' '.join(reduce_lengthening(eachWord) for eachWord in x if eachWord not in stopwordsList))
        else:
            if len(listOfStopWords) > 0 or append_stopwords:
                raise CustomException("Please set the remove_stopwords flag to True or set append_stopwords = False and listOfStopWords = []")
            data[new_column_name] = data['tokens'].apply(lambda x: ' '.join(reduce_lengthening(eachWord) for eachWord in x))

    
    data[new_column_name] = data[new_column_name].apply(lambda x: re.sub(r"'s\b","",x))
    data[new_column_name] = data[new_column_name].apply(lambda x: re.sub('\s+', ' ', x.strip()))
    data.drop(columns=['temporary', 'tokens'], inplace = True, axis = 1)

    return data

# Concatinate strings in different rows
def concatString(elements):
	return ' '.join(str(ele) for ele in elements)

# Creating words clouds for n-grams (n=2,3...)
def createNgramWordCloud(dictionaryKeys, 
                        dictValues, 
                        n, 
                        remove_stopwords, 
                        save_fig):
    newDictKeys = [concatString(x).strip() for x in dictionaryKeys]
    newDictionary = dict(zip(newDictKeys, dictValues))

    if remove_stopwords:
        stopwords = set(STOPWORDS)
        wordCloudbiGram = WordCloud(width = n*1000, height = n*1000, 
                    background_color ='white', stopwords = stopwords,
                    min_font_size = 20).generate_from_frequencies(newDictionary)
    else:
        wordCloudbiGram = WordCloud(width = n*1000, height = n*1000, 
                    background_color ='white',
                    min_font_size = 20).generate_from_frequencies(newDictionary)
    plt.imshow(wordCloudbiGram)    
    if save_fig:
        plt.savefig(str(n)+'_gram.png')
    plt.show()

# Creating unigram word clouds 
def createUnigramWordCloud(corpus, 
                        remove_stopwords, 
                        save_fig):

    if remove_stopwords:
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(width = 1000, height = 1000, 
                    background_color ='white', stopwords = stopwords,
                    min_font_size = 20).generate(corpus)
    else:
        wordcloud = WordCloud(width = 1000, height = 1000, 
                    background_color ='white',
                    min_font_size = 20).generate(corpus)
    
    plt.imshow(wordcloud)
    if save_fig:
        plt.savefig('Unigram.png')

    plt.show()

# Create n-grams
# n = 2,3 for bi grams, trigrams respectively 
"""
    Create unigram, bigram or n-gram wordclouds

    :data: Pandas dataframe containing the text
    :column_name: Name of column containing the text
    :remove_stopwords: True/False Filter stop words from wordcloud
    :n: Value of n in n-gram
    :save_fig: True/False to save the chart
    - return: None
"""
def create_word_cloud(data, column_name, remove_stopwords = True, n = 1, save_fig = False):
    frequencies = Counter([])
    corpus = data[column_name].str.cat()
    if n == 1:
        createUnigramWordCloud(corpus, remove_stopwords, save_fig)
    else:
        token = word_tokenize(corpus)
        ngramsCreated = ngrams(token, n)
        frequencies += Counter(ngramsCreated)
        dictionaryValues = dict(frequencies)
        dictionaryKeys = dictionaryValues.keys()
        dictValues = dictionaryValues.values()
        createNgramWordCloud(dictionaryKeys, dictValues, n, remove_stopwords, save_fig)
