import json

import gensim
import numpy as np
from nltk.tokenize import word_tokenize

'''
Embeder takes canopies as input
and return a 2d np array with each row being the vector representation of the canopy
'''
class BaseEmbeder(object):

    def __init__(self, args):
        self.args = args
        self.embeddings = []

    '''
    This function should overwrite for different embeders
    '''
    def embed(self, canopies):
        pass


class GloveEmbeder(BaseEmbeder):

    def embed(self, canopies):
        glove_loc = 'embedding/glove/glove.6B.300d_word2vec.txt'
        self.embeddings = []
        self.embed_dims = 300
        self.model = gensim.models.KeyedVectors.load_word2vec_format(glove_loc, binary=False)

        for canopy in canopies:
            sub = canopy[0].sub
            if sub in self.model.vocab:
                assert(np.any(self.model.word_vec(sub)) == True)
                self.embeddings.append(self.model.word_vec(sub))
            else:
                vec = np.zeros(self.embed_dims, np.float32)
                wrds = word_tokenize(sub)
                count = 0
                for wrd in wrds:
                    if wrd in self.model.vocab:
                        count += 1
                        vec += self.model.word_vec(wrd)

                if count != 0:
                    assert(np.any(vec/count) == True)
                    self.embeddings.append(vec/count)
                else:
                    assert(np.any(self._embed_unknown(canopy)) == True)
                    self.embeddings.append(self._embed_unknown(canopy))

        self.embeddings = np.array(self.embeddings)
        return self.embeddings

    def _embed_unknown(self, canopy):
        u_embedding = np.zeros(self.embed_dims, np.float32)
        count = 0
        for mention in canopy:
            obj = mention.obj
            if obj in self.model.vocab:
                u_embedding += self.model.word_vec(obj)
                count += 1
            else:
                vec = np.zeros(self.embed_dims, np.float32)
                wrds = word_tokenize(obj)
                tmp = 0
                for wrd in wrds:
                    for wrd in self.model.vocab:
                        tmp += 1
                        vec += self.model.word_vec(wrd)
                    if tmp != 0:
                        u_embedding += vec / tmp
                        count += 1
        if count != 0:
            return u_embedding / count
        else:
            return np.random.randn(self.embed_dims)


class BertEmbeder(BaseEmbeder):

    def load_bert_encode(self):
        fpath = 'embedding/bert/{}_{}_bert_encode.json'.format(self.args.dataset, self.args.split)
        with open(fpath) as f:
            self.bert_encode = json.load(f)

    def embed(self, canopies):
        self.embeddings = []
        self.bert_dim = 768
        for canopy in canopies:
            canopy_vec = np.zeros(self.bert_dim)
            for mention in canopy:
                if str(mention.id) not in self.bert_encode:
                    bert_vec = np.random.randn(self.bert_dim)
                else:
                    bert_vec = np.array(self.bert_encode[str(mention.id)]['sub_vec'])
                canopy_vec += bert_vec
            if len(canopy) == 0:
                print('empty canopy')
                canopy_vec = np.random.randn(self.bert_dim)
            else:
                canopy_vec /= len(canopy)
            self.embeddings.append(canopy_vec)
        return np.array(self.embeddings)

class ElmoEmbeder(BaseEmbeder):

    def load_elmo_encode(self):
        pass

    def embed(self, canopies):
        pass
