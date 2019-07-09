import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import logging
import numpy as np
import string
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
from sklearn.ensemble import ExtraTreesClassifier
from collections import defaultdict


import embeddingvectorizer
from nltk.corpus import stopwords

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

path_to_data ='../data/'
df = pd.read_pickle(path_to_data + "data_geannoteerd.pkl")
data = df['text']
labels = df['topic']

basepath = '/home/anne/tmpanne/fullsample/test_'

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.02, random_state=42)


class word2vec_analyzer():
    '''This class tests the efficacy of Word2Vec models in downstream tasks.'''

    def __init__(self):
        self.nmodel = 0
        self.vectorizer = 'Tfidf'
        
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
        
    def get_best_parameters_w2v(self, model):
        
        if self.vectorizer =='Tfidf':
            logger.info("Starting gridsearch to optimize parameter setting for model {} using TfidfVectorizer ...".format(model['filename']))

            pipeline = Pipeline([("word2vec Tfidf vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(model, operator='mean')),
                                 ("clf", SGDClassifier(loss='hinge', tol=1e-4))
                ])

        else:
            logger.info("Starting gridsearch to optimize parameter setting for model {} using CountVectorizer...".format(model['filename']))
            pipeline = Pipeline([("word2vec Count vectorizer", embeddingvectorizer.EmbeddingCountVectorizer(model, operator='mean')),
                             ("clf", SGDClassifier(loss='hinge', tol=1e-4))
            ])


        param_grid =  {'clf__max_iter': (20, 30) , 'clf__alpha': (0.00001, 0.000001), 'clf__penalty': ('l2', 'elasticnet')}


        search = GridSearchCV(pipeline, param_grid, iid=False, cv=5)
        logger.info("Start GridSearch for model {} ...".format(model['filename']))

        search.fit(X_train, y_train )   

        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print()
        print("Best parameters:", search.best_params_)

        return search.best_params_
    
    
    def apply_bestparameters_w2v(self, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test):
        results = []
        
        for model in self.get_w2v_model():
            
            logger.info(">>>> Retrieving best parameter settings for the model {} ...".format(model['filename']))
            bp = self.get_best_parameters_w2v(model)

            if self.vectorizer == "Tfidf":
                logger.info("Apply best parameter setting for the model {} ...".format(model['filename']))

                pipeline = Pipeline([("word2vec Tfidf vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(model, operator='mean')),
                                     ("clf", SGDClassifier(loss='hinge', tol=1e-4, penalty=bp['clf__penalty'], alpha=bp['clf__alpha'], max_iter=bp['clf__max_iter']))
                    ])

            else:
                logger.info("Apply best parameter setting for the model {} ...".format(model['filename']))
                pipeline = Pipeline([
                    ("word2vec Count vectorizer", embeddingvectorizer.EmbeddingCountVectorizer(model, operator='mean')),
                    ("clf", SGDClassifier(loss='hinge', tol=1e-4, penalty=bp['clf__penalty'], alpha=bp['clf__alpha'], max_iter=bp['clf__max_iter']))
                ])


        clf = pipeline.fit(X_train, y_train)   
        logger.info("fitted...{} ...".format(model['filename']))

        predicted = clf.predict(X_test)
        logger.info("predicted...{} ...".format(model['filename']))

        precision, recall, fscore, support= score(y_test, predicted, average='macro')

        results.append({'precision': precision, 
                'recall': recall, 
                'f1': fscore, 
                'classifier' : "SGD", 
                'penalty' : bp['clf__penalty'], 
                'alpha' : bp['clf__alpha'], 
                'max_iter' : bp['clf__max_iter'] , 
                'vectorizer' : vect , 
                'model' : model['filename']})

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

    df = pd.DataFrame.from_dict(gensimscore)
    print('Created dataframe')
    # print(df)
    df.to_csv('w2v_evaluation.csv')

            