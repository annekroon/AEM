import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import logging
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn import svm
from datetime import datetime
import os

from gensim.models.word2vec import Word2Vec
import gensim

from sklearn.pipeline import Pipeline
from collections import defaultdict

import embeddingvectorizer
from nltk.corpus import stopwords

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfTransformer
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path_to_data ='../data/'
df = pd.read_pickle(path_to_data + "data_geannoteerd.pkl")
data = df['text']
labels = df['topic']

basepath = '/home/anne/tmpanne/fullsample/'

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.02, random_state=42)

class word2vec_analyzer():
    '''This class tests the efficacy of Word2Vec models in downstream tasks.'''

    def __init__(self):
        self.nmodel = 0
        self.vectorizer = 'Count'
        self.param_grid = {'clf__penalty': ('l2', 'elasticnet') }
        
       # logger.info("Created analyzer with {} combinations for crime and {} combinations for low life".format(
       #     len(self.combinations_crime), len(self.combinations_low)))

    def get_w2v_model(self):
        '''yields a dict with one item. key is the filename, value the gensim model'''
        
        filenames = [e for e in os.listdir(basepath) if not e.startswith('.')]

        for fname in filenames:
            model = {}
            path = os.path.join(basepath, fname)
            logger.info("\nLoading gensim model")
            mod = gensim.models.Word2Vec.load(path)
            model['gensimmodel'] = dict(zip(mod.wv.index2word, mod.wv.syn0))
            model['filename'] = fname
            #splitResult = fname.split( "_" ) #split on scores
            self.nmodel +=1
            logger.info("loaded gensim model nr {}, named: {}".format(self.nmodel, model['filename']))
            yield model
        
    def get_best_parameters(self):
        
        results = []
        
        if self.vectorizer == 'Tfidf':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(loss='hinge', tol=1e-4, alpha=1e-6, max_iter=1000, random_state=42)),
            ])
            
        else:
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('clf', SGDClassifier(loss='hinge', tol=1e-4, alpha=1e-6, max_iter=1000, random_state=42)),
            ])
            
        param_grid = self.param_grid

        search = GridSearchCV(pipeline, param_grid, iid=False, cv=5)
        logger.info("Start GridSearch ...")

        search.fit(X_train, y_train )   
      
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print()
        print("Best parameters:", search.best_params_)
        

        return search.best_params_ 
    
  
    def apply_bestparameters_w2v(self, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
        
 #       logger.info("retrieved the best parameter settings: {}...  ".format(bp))
#        logger.info("\n\n\nthese are the results of the baseline model with hyperparameter optimalization: {}".format(results_baseline))
        
        results = []
        
        bp = self.get_best_parameters()
        
        if self.vectorizer == "Tfidf":
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', SGDClassifier(loss='hinge', alpha=1e-6, tol=1e-4, max_iter=1000, random_state=42, penalty=bp['clf__penalty'])
                 )])
            
        else:
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('clf', SGDClassifier(loss='hinge', alpha=1e-6, tol=1e-4, max_iter=1000, random_state=42, penalty=bp['clf__penalty'])
                 )])
            
        clf = pipeline.fit(X_train, y_train)   
        logger.info("fitted baseline model.")

        test_pred = clf.predict(X_test)
        logger.info("predicted baseline model.")

        accuracy = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, average = 'macro')
        recall = recall_score(y_test, test_pred, average = 'macro')
        f1score = f1_score(y_test, test_pred, average = 'macro')

        print("accurcay {}, precision {} recall {} and f1score {} baseline".format(accuracy, precision, recall, f1score))
        
        results.append({'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1score': f1_score, 
            'classifier' : "SGD",
            'model' : 'baseline'})
        
        
        for model in self.get_w2v_model():
            logger.info(">>>> Retrieving best parameter settings for the model {} ...".format(model['filename']))

            
            if self.vectorizer == "Tfidf":
                
                logger.info("Apply best parameter setting for the model {} ...".format(model['filename']))
                pipeline = Pipeline([
                    ("word2vec Tfidf vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(model['gensimmodel'], operator='mean')),
                    ("clf", SGDClassifier(loss='hinge', alpha=1e-6, tol=1e-4, max_iter=1000, random_state=42, penalty=bp['clf__penalty']))
                    ])
                
                
                
            else:
                logger.info("Apply best parameter setting for the model {} ...".format(model['filename']))
                pipeline = Pipeline([
                    ("word2vec Count vectorizer", embeddingvectorizer.EmbeddingCountVectorizer(model['gensimmodel'], operator='mean')),
                    ("clf", SGDClassifier(loss='hinge', alpha=1e-6, tol=1e-4, max_iter=1000, random_state=42, penalty=bp['clf__penalty']))
                    ])
                
                
            clf = pipeline.fit(X_train, y_train)   
            logger.info("fitted...{} ...".format(model['filename']))

            test_pred = clf.predict(X_test)
            logger.info("predicted...{} ...".format(model['filename']))

            accuracy = accuracy_score(y_test, test_pred)
            precision = precision_score(y_test, test_pred, average = 'macro')
            recall = recall_score(y_test, test_pred, average = 'macro')
            f1score = f1_score(y_test, test_pred, average = 'macro')

            results.append({'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1score': f1_score, 
            'classifier' : "SGD",
            'model' : model['filename']})

            print("these are the results of the w2v models:", accuracy, precision, recall)
            
        return results
        
if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    myanalyzer = word2vec_analyzer()
    my_results = myanalyzer.apply_bestparameters_w2v()
    
    print("\n\n\nSave results\n\n\n")
    with open('output_my_results.json',mode='w') as fo:
        fo.write('[')
        
        for result in my_results:
            #print("this is the result:", result)
            fo.write(json.dumps(result))
            fo.write(',\n')
        fo.write('[]]')

    df = pd.DataFrame.from_dict(my_results)
    print('Created dataframe')
    # print(df)
    df.to_csv('w2v_evaluation.csv')


            
