from helper import *
from unionFind import DisjointSet
from nltk.corpus import stopwords

import math
import itertools
import editdistance

class Cleaner():

    def __init__(self, args, triples_list, amb_ent, is_acronym, true_clust2ent):
        self.p = args
        self.triples_list = triples_list
        self.amb_ent = amb_ent
        self.is_acronym = is_acronym
        self.true_clust2ent = true_clust2ent
        if 'reverb45k' in self.p.dataset:
            self.fix_typos()
        self.fix_typos()
        self.process()

    def fix_typos(self):

        uf = DisjointSet()

        '''
        use wiki info to correct typos
        Very high confidence
        '''
        ent2wiki = ddict(set)
        for trp in self.triples_list:
            [sub, rel, obj] = trp['triple']

            if trp['ent_lnk_sub'] != None: ent2wiki[sub].add(trp['ent_lnk_sub'])
            if trp['ent_lnk_obj'] != None: ent2wiki[obj].add(trp['ent_lnk_obj'])

        wiki2ent = invert_dict(ent2wiki, 'm2os')
        for wiki, clust in wiki2ent.items():
            for e1, e2 in itertools.combinations(clust, 2):
                if e1 == e2: continue
                if e1 == '' or e2 == '':
                    print('Missing wiki linking')
                    continue
                if e1 in self.amb_ent or e2 in self.amb_ent: continue

                if editdistance.eval(e1, e2) <= 2:
                    uf.add(e1, e2)
                elif e1 in self.is_acronym or e2 in self.is_acronym:
                    print('acronym')
                    uf.add(e1, e2)

        ''' fix typos '''
        ent2rep = {}
        for rep, clust in uf.group.items():
            rep = max(clust, key=len)
            for ele in clust:
                ent2rep[ele] = rep

        for i, trp in enumerate(self.triples_list):
            [sub, rel, obj] = trp['triple']

            if sub in ent2rep:
                new_sub = ent2rep[sub]
            else:
                new_sub = ' '.join([ent2rep.get(ele, ele) for ele in sub.split()])

            self.triples_list[i]['triple_fixed'] = [new_sub, rel, obj]

        rep2ent = invert_dict(ent2rep, 'm2o')
        self._print_typo_fixed(rep2ent)

    # method aborted
    def tf_idf(self):
        N = 0
        TF = ddict(int)
        Nx = ddict(int)
        TF_IDF_dict = {}
        stop_words = stopwords.words('english')
        for trp in self.triples_list:
            N += 1
            tok_set = set()
            for item in trp['triple_fixed']:
                for tok in item.split():
                    if tok not in stop_words:
                        tok_set.add(tok)
                        TF[tok] += 1
            for tok in tok_set:
                Nx[tok] += 1
        max_tf = 0
        for k, v in TF.items():
            if v > max_tf: max_tf = v
        for k, v in TF.items():
            TF_IDF_dict[k] = (v/max_tf) * (math.log((N+1)/(Nx[k]+1)))
        for trp in self.triples_list:
            sub = trp['triple_fixed'][0]
            max_idf_score = max([TF_IDF_dict[tok] for tok in sub.split()])
            for tok in sub.split():
                if tok in stop_words:
                    print('{} is stop words in {}'.format(tok, sub))
                    sub = sub.replace(tok,'')
                    continue
                if TF_IDF_dict[tok] < max_idf_score / 10:
                    print('{} has relative small idf score than {}'.format(tok, sub))
                    sub = sub.replace(tok,'')
            trp['triple_fixed'][0] = sub


    def process(self):
        self.sub2trp = ddict(list)
        for trp in self.triples_list:
            sub = trp['triple_fixed'][0]
            self.sub2trp[sub].append(trp)

        self.sub_list = list(self.sub2trp.keys())

    def get_return(self):
        return self.sub_list, self.sub2trp, self.triples_list

    def _print_typo_fixed(self, rep2ent):
        fn = self.p.out_path + '/typo_fixed.txt'
        with open(fn, 'w') as f:
            for k, v in rep2ent.items():
                f.write('{}: {}\n'.format(k, v))
