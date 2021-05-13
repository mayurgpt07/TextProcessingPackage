# TextProcessingPackage

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/preethampaul/TextProcessingPAckage/blob/master/LICENSE) </br>
Most of the text based problems require a lot of pre processing. </br>
The package is an attempt to reduce the time of preprocessing so that the user can focus on developing solutions to problems

The package contains the following techniques:
* Removal of unenecessary punctuations
* Removal of html/xml tags
* Removal of stop words (optional)
* Removal of digits (optional)
* Expanding short forms (aren't -> are not)
* Lemmatization (optional)
* Creation of word clouds with combinaton of n words (n-grams)
* Addition of words to stopwords list

## Installation
```
pip install TextPreProcessing
```

## Code Snippets
#### Preprocessing of text without lemmatization

```
from DataAndProcessing import text_cleaner
dataframe = text_cleaner(data, column_name='column_with_textdata', remove_stopwords=True, listOfStopWords = ['no','none'], append_stopwords=True, remove_digits=False, do_lemmatization=True)
```

#### Create Word Cloud

```
from DataAndProcessing import create_word_cloud
create_word_cloud(dataframe, column_name='column_with_textdata', n=1, save_fig=False)
```
where n = 1,2,3 means unigram, bigram and trigram respectively

#### Append words to the standard NLTK stopwords list
```
from DataAndProcessing import add_stopwords
add_stopwords(listOfStopWords, is_new = False)
```
