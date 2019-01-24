#!/usr/bin/env python3
import inca
import gensim
import logging
import re

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')

PATH = "/home/anne/tmpanne/"
FILENAME = "uniekezinnen_2000-01-01_2016-12-31"
outlets = ['telegraaf (print)', 'nrc (print)', 'volkskrant (print)', 'ad (print)', 'trouw (print)',  'telegraaf (www)', 'nrc (www)', 'volkskrant (www)', 'ad (www)', 'trouw (www)', 'nu' , 'nos']

w2v_params = {
    'alpha': 0.025,
    'size': 100,
   # 'window': 15,
   # 'iter': 5,
    'min_count': 5,
    'sg': 1,
    'hs': 0,
    'negative': 5
}

def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self, doctype, fromdate,todate):
        self.doctype = doctype
        self.fromdate = fromdate
        self.todate = todate
      
        self.model = gensim.models.Word2Vec(**w2v_params)
        self.model.build_vocab(self.get_sentences_vocab())
        print('Build Word2Vec vocabulary')
        self.model.train(self.get_sentences_train(),total_examples=self.model.corpus_count, epochs=self.model.iter)
        print('Estimated Word2Vec model')



    def get_sentences_vocab(self):
        with open(PATH+FILENAME) as fi:
            for line in fi:
                sentence = line.split()
                if len(sentence)>0:
                    yield sentence



    def get_sentences_train(self):
        with open(PATH+FILENAME) as fi:
            for line in fi:
                sentence = line.split()
                if len(sentence)>0:
                    yield sentence


def train_and_save(fromdate,todate,doctype):
    filename = "{}w2v_all__uniquesentences_{}_{}".format(PATH,fromdate,todate)

    casus = train_model(doctype,fromdate,todate)

    with open(filename, mode='wb') as fo:
        casus.model.save(fo)
    print('Saved model')
    print("reopen it with m = gensim.models.FastText.load('{}')".format(filename))
    del(casus)


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save(fromdate = "2000-01-01", todate = "2016-12-31", doctype = outlets)
