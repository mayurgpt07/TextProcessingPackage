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

Currently in work
* Topic Modeling using LDA and Non Negative Matrix factorizations using both sklearn and gensim

## Installation
```
pip install TextPreProc
```

## Code Snippets
#### Preprocessing of text with lemmatization

```
from TextProcessing.DataAndProcessing import text_cleaner
dataframe = text_cleaner(dataframe, 
                         column_name='column_with_textdata', 
                         remove_stopwords=True, 
                         listOfStopWords = ['no','none'], 
                         append_stopwords=True, 
                         remove_digits=False, 
                         do_lemmatization=True)
```
Output
* Return a datafram with a new column ('column_with_textdata' + '_processed')

Parameters and Values
* remove_stopwords: True or False (Default: True)
* listOfStopWords: list of stop words to use remove from data (Dafault: empty list, Requires: remove_stopwords = true)
* append_stopwords: True or False (Default: True)
* remove_digits: True or False (Default: False)
* do_lemmatization: True or False (Default: True)

#### Create Word Cloud

```
from TextProcessing.DataAndProcessing import create_word_cloud
create_word_cloud(dataframe, column_name='column_with_textdata', n=1, save_fig=False)
```
where n = 1,2,3 means unigram, bigram and trigram respectively

#### Append words to the standard NLTK stopwords list
```
from TextProcessing.DataAndProcessing import add_stopwords
add_stopwords(listOfStopWords, is_new = False)
```
### Topic Modelling
```
from TextProcessing.TopicModelling import Topic_Modelling
tm = Topic_Modelling(dataframe, 
                     column_name='response_text', 
                     vectorizer_type = 'bow', 
                     topic_modelling_type = 'lda', 
                     vectorizer_parameters = {'strip_accents': 'unicode'}, 
                     topic_model_parameters = {'init': 'random'}, 
                     num_topics = 10, 
                     show_visualization = True, 
                     save_fig = False, 
                     fig_title = 'Topics in the data', 
                     is_sklearn = True)
tm.fit_transform()
```
Output
* None (Currently the model does not produce any output)
Parameters and values
* **vectorzier_type**: 'bow' or 'tfidf' (Default: 'bow')
* **topic_modelling_type**: 'lda' or 'nmf' (Default: 'lda')
* **vectorizer_parameters**: dictionary of parameters for sklearn CountVectorizer or TFIDFVectorizer (Default: Default model parameters)
* **topic_model_parameters**: dictionary of parameters for sklearn NMF or LatentDirichletAllocation (Default: Default model parameters)
* **num_topics**: number of topics (Default: 10)
* **show_visualization**: True or False (Default: True)
* **save_fig**: True or False (Default: False)
* **fig_title**: Title for the plot (Default: 'Topics in the data')
* **is_sklearn**: True or False (Currently the module supports on Sklearn modules) 

### Run only the vectorizer and features as values
```
tm.return_models(get_results = True, drop_text_columns = True)
```
Output
* Returns dataframe with vectorizer features as columns
* Returns vectorizer chosen

Parameters and values
* **get_results** = Return the dataset with vectorizer features as columns (Default: True)
* **drop_text_columns** = Delete the base text column (Default: False)

### Return the trained vectorizer and topic models
```
tm.return_models(return_vectorizer = True, return_topic_model = True)
```
Output
* Returns dictionary of vectorizer and topic models 

Parameters and values
* **return_vectorizer** = True or False (Default: False)
* **return_topic_model** = True or False (Default: True)

### Example of cleaning data and topic modelling with TF-IDF and Non-Negative Matrix Factorization
![alt text](SampleCode.png)
