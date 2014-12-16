# Sebastian Raschka 2014

import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  
from sklearn.grid_search import GridSearchCV  
from time import time  
from sklearn.pipeline import Pipeline  
from sklearn.naive_bayes import MultinomialNB, BernoulliNB  
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import EnglishStemmer
import pickle
from sklearn import cross_validation


master_dir = os.path.dirname(os.path.realpath(__file__))

###########################################
# Setting up tokenizer
###########################################



stop_words = pickle.load(open(os.path.join(master_dir, '../stopwords.p'), 'rb'))
semantic_words = pickle.load((open(os.path.join(master_dir, '../whitelist/semantic_words.p'), 'rb')))

porter = PorterStemmer()
snowball = EnglishStemmer()

# raw words
# tokenizer = lambda text: text.split()
def tokenizer(text):
    return text.split()

# words after Porter stemming 
# tokenizer_porter = lambda text: [porter.stem(word) for word in text.split()]
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Words after Snowball stemming
# tokenizer_snowball = lambda text: [snowball.stem(word) for word in text.split()]
def tokenizer_snowball(text):
    return [snowball.stem(word) for word in text.split()] 

# Only words that are in a list of 'positive' or 'negative' words ('whitelist')
# http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
#tokenizer_whitelist = lambda text: [word for word in text.split() if word in semantic_words]
def tokenizer_whitelist(text):
    return [word for word in text.split() if word in semantic_words]

# Porter-stemmed words in whitelist
# tokenizer_porter_wl = lambda text: [porter.stem(word) for word in text.split() if word in semantic_words]
def tokenizer_porter_wl(text):
    return [porter.stem(word) for word in text.split() if word in semantic_words]

# Snowball-stemmed words in whitelist
# tokenizer_snowball_wl = lambda text: [snowball.stem(word) for word in text.split() if word in semantic_words]
def tokenizer_snowball_wl(text):
    return [snowball.stem(word) for word in text.split() if word in semantic_words]



###########################################
# Loading training data
###########################################
df_train = pd.read_csv(os.path.join(master_dir, '../../data/labeledTrainData.tsv'), sep='\t', quoting=3)
df_test = pd.read_csv(os.path.join(master_dir, '../../data/testData.tsv'), sep='\t', quoting=3)
X_train = df_train['review']
y_train = df_train['sentiment']


###########################################
# Pipeline of feature extractor, classifier, and parameters
###########################################
pipeline_ber = Pipeline([
                         ('vec', CountVectorizer(binary=True)),
                         ('clf', BernoulliNB())])

parameters_ber = {  
'vec__tokenizer': (tokenizer, tokenizer_porter, tokenizer_snowball, 
                   tokenizer_whitelist, tokenizer_porter_wl, tokenizer_snowball_wl),
'vec__max_df': (0.5, 0.75, 1.0),  
'vec__max_features': (None, 5000),  
'vec__min_df': (1, 50),  
'vec__stop_words': [None, stop_words],
'vec__ngram_range' : [(1,1), (1,2), (2,2)],}



###########################################
## Run GridSearch
###########################################
grid_search = GridSearchCV(pipeline_ber, 
                           parameters_ber, 
                           n_jobs=1, 
                           cv=3, 
                           scoring='roc_auc',
                           verbose=2)  
t0 = time()

print('Start Gridsearch')

grid_search.fit(X_train, y_train)  


print('\n\n\n\n{0}\nREPORT\n{0} '.format(50*'#'))
print('done in {0}s'.format(time() - t0))  
print('Best score: {0}'.format(grid_search.best_score_))  
print('Best parameters set:')  
best_parameters = grid_search.best_estimator_.get_params()  
for param_name in sorted(list(parameters.keys())):  
    print('\t{0}: {1}'.format(param_name, best_parameters[param_name]))
print('\n\n\n All Scores:')
print(grid_search.grid_scores_)
