#!/usr/bin/env python3
import inca
import gensim
import logging
import re

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')

PATH = "/home/anne/tmpanne/"
#outlets = ['telegraaf (print)', 'nrc (print)', 'volkskrant (print)', 'ad (print)', 'trouw (print)']
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
        if type(self.doctype) is str:
            self.query = {
                  "query": {
                          "bool": {
                                    "filter": [
                                                { "match": { "doctype": self.doctype}},
                                                { "range": { "publication_date": { "gte": self.fromdate, "lt":self.todate }}}
                                              ]
                                  }
                        }
                }
        elif type(self.doctype) is list:
            self.query = {
                  "query": {
                          "bool": {
                                    "filter": [ {'bool': {'should': [{ "match": { "doctype": d}} for d in self.doctype]}},
                                                { "range": { "publication_date": { "gte": self.fromdate, "lt":self.todate }}}
                                              ]
                                  }
                        }
                }


        self.documents = 0
        self.failed_document_reads = 0
        self.model = gensim.models.Word2Vec(**w2v_params)

        self.sentences = set()
        for sentence in self.get_sentences():
            self.sentences.add(" ".join(sentence))



    def get_sentences(self):
        for d in inca.core.database.scroll_query(self.query):
            try:
                self.documents += 1
                sentences_as_lists = (s.split() for s in preprocess(d['_source']['text']).split('.'))
                for sentence in sentences_as_lists:
                    yield sentence
            except:
                self.failed_document_reads +=1
                continue

def train_and_save(fromdate,todate,doctype):
    filename = "{}uniekezinnen_{}_{}".format(PATH,fromdate,todate)

    casus = train_model(doctype,fromdate,todate)

    with open(filename, mode='w') as fo:
        for sentence in casus.sentences:
            fo.write(sentence)
            fo.write('\n')
    
    print('Created file with sentences: {}'.format(filename))


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save(fromdate = "2000-01-01", todate = "2016-12-31", doctype = outlets)
