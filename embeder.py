import numpy as np
import gensim, json

from nltk.tokenize import word_tokenize

class Embeder():

    def __init__(self, sub_list, sub2trp, triples_list):
        self.sub2embed = {}
        self.sub_list = sub_list
        self.sub2trp = sub2trp
        self.triples_list = triples_list
        self.glove_encode = np.array([])
        self.get_glove_embedding()
        self.get_bert_embedding()

    def get_glove_embedding(self):
        glove_loc = 'glove/glove.6B.300d_word2vec.txt'
        embed_dims = 300
        model = gensim.models.KeyedVectors.load_word2vec_format(glove_loc, binary=False)

        embed_list = []
        for sub in self.sub_list:
            if sub in model.vocab:
                embed_list.append(model.word_vec(sub))
            else:
                vec = np.zeros(embed_dims, np.float32)
                wrds = word_tokenize(sub)
                for wrd in wrds:
                    count = 0
                    if wrd in model.vocab:
                        count += 1
                        vec += model.word_vec(wrd)

                if count != 0:    embed_list.append(vec / count)
                else:
                    #print('Uknown sub: {}'.format(sub))
                    embed_list.append(self._embed_unknown(sub, model))

        self.glove_encode = np.array(embed_list)

    '''
    use obj to represent unknown subjects
    '''
    def _embed_unknown(self, sub, model):
        embed_dims = 300
        embed = np.zeros(embed_dims, np.float32)
        count = 0
        for trp in self.sub2trp[sub]:
            obj = trp['triple_fixed'][2]
            if obj in model.vocab:
                embed += model.word_vec(obj)
                count += 1
            else:
                vec = np.zeros(embed_dims, np.float32)
                wrds = word_tokenize(obj)
                for wrd in wrds:
                    tmp = 0
                    if wrd in model.vocab:
                        tmp += 1
                        vec += model.word_vec(wrd)
                    if tmp != 0:
                        embed += vec / tmp
                        count += 1

        if count != 0:    return embed / count
        else:
            #print('Random Initial: {}'.format(sub))
            return np.random.randn(embed_dims)

    def get_bert_embedding(self):
        #fn = 'bert/bert_encode_reverb.json'
        #fn = 'bert/bert_encode_ambiguous_cesi.json'
        fn = 'bert/bert_difficulty_test.json'
        with open(fn) as f:
            bert_encode = json.load(f)
        for trp in self.triples_list:
            tmp = bert_encode[str(trp['id'])]
            trp['sub_bert_vec'] = np.array(tmp['sub_vec'])
            trp['rel_bert_vec'] = np.array(tmp['rel_vec'])
            trp['obj_bert_vec'] = np.array(tmp['obj_vec'])

    def get_return(self):
        return self.glove_encode, self.triples_list
