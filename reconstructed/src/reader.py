import json
import itertools
import editdistance
from collections import defaultdict as ddict

import codecs
from nltk.corpus import stopwords

from structure.mention import Mention
from structure.sideInfo import SideInfo
from utils.helper import invert_dict
from utils.unionFind import DisjointSet
from utils.printer import *

class BaseReader(object):

    def __init__(self, args):
        self.args = args
        self.fpath = '../datasets/{}/{}_{}'.format(args.dataset, args.dataset, args.split)
        self.mentions_list = []

    # read data
    # override for different datasets
    def process(self):
        pass

    def get_ground_truth(self):
        # set label
        if self.args.label == 'wiki':
            for mention in self.mentions_list:
                mention.label = mention.side_info.wiki_link_sub
        elif self.args.label == 'freebase':
            for mention in self.mentions_list:
                mention.label = mention.side_info.freebase_link_sub

        # set omit
        if self.args.omit:
            for mention in self.mentions_list:
                if mention.label is None: mention.omit = True
                else: mention.omit = False
        else:
            for mention in self.mentions_list:
                mention.omit = False

        # get ground truth
        self.true_ent2clust = ddict(set)
        for mention in self.mentions_list:
            if mention.omit: continue
            self.true_ent2clust[mention.sub_u].add(mention.label)
        self.true_clust2ent = invert_dict(self.true_ent2clust, 'm2os')

        gt_path = self.args.out_path + '/ground_truth'
        print_ground_truth(self.args.label, self.mentions_list, self.true_clust2ent, gt_path)

    def get_ambiguous_ent(self):
        self.amb_clust = {}
        self.acronym = {}
        for mention in self.mentions_list:
            if mention.sub.isalpha() and mention.sub.isupper():
                self.acronym[mention.sub] = 1
            if mention.obj.isalpha() and mention.obj.isupper():
                self.acronym[mention.obj] = 1
            for tok in mention.sub.split():
                self.amb_clust[tok] = self.amb_clust.get(tok, set())
                self.amb_clust[tok].add(mention.sub)

        self.amb_ent = ddict(int)
        self.amb_mentions = {}
        for rep, clust in self.amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1

    def fix_typo(self):
        self.get_ambiguous_ent()
        uf = DisjointSet()

        #clean with wiki link
        ent2wiki = ddict(set)
        for mention in self.mentions_list:
            if mention.side_info.wiki_link_sub is not None:
                ent2wiki[mention.sub].add(mention.side_info.wiki_link_sub)
            if mention.side_info.wiki_link_obj is not None:
                ent2wiki[mention.obj].add(mention.side_info.wiki_link_obj)

        wiki2ent = invert_dict(ent2wiki, 'm2os')
        for wiki, clust in wiki2ent.items():
            for e1, e2 in itertools.combinations(clust,2):
                if e1 == e2: continue
                if e1 == '' or e2 == '': continue
                if e1 in self.amb_ent or e2 in self.amb_ent: continue

                if editdistance.eval(e1, e2) <= 2:
                    uf.add(e1, e2)
                elif e1 in self.acronym or e2 in self.acronym:
                    uf.add(e1, e2)

        #fix
        ent2rep = {}
        for rep, clust in uf.group.items():
            rep = max(clust, key=len)
            for ele in clust:
                ent2rep[ele] = rep

        for i, mention in enumerate(self.mentions_list):
            if mention.sub in ent2rep:
                new_sub = ent2rep[mention.sub]
            else:
                new_sub = ' '.join([ent2rep.get(ele, ele) for ele in mention.sub.split()])

            if mention.sub != new_sub:
                #print('fix {} to {}'.format(mention.sub, new_sub))
                self.mentions_list[i].sub = new_sub

    def get_return(self):
        return self.mentions_list, self.true_ent2clust, self.true_clust2ent


class AmbiguousReader(BaseReader):

    def process(self):
        with codecs.open(self.fpath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                raw_data = json.loads(line.strip())

                sub, rel, obj = json.loads(line.strip())
                if len(sub.strip()) == 0 or len(rel.strip()) == 0 or len(obj.strip()) == 0:
                    print('Incomplete mention ({}, {}, {})'.format(sub, rel, obj))
                    continue

                id = raw_data['_id']
                origin_triple = raw_data['triple']
                freebase_link = raw_data['true_link']
                wiki_link = raw_data['entity_linking']
                src_sentences = raw_data['src_sentences']

                side_info = SideInfo(origin_triple, freebase_link, wiki_link, src_sentences)
                mention = Mention(sub, rel, obj, id, side_info)

                self.mentions_list.append(mention)
        self.fix_typo()
        self.get_ground_truth()


class Reverb45kReader(BaseReader):

    def process(self):
        # read from data files
        with codecs.open(self.fpath, encoding='utf-8', errors='ignore') as f:
            for line in f:
                raw_data = json.loads(line.strip())

                sub, rel, obj = map(str, raw_data['triple_norm'])
                if len(sub.strip()) == 0 or len(rel.strip()) == 0 or len(obj.strip()) == 0:
                    print('Incomplete mention ({}, {}, {})'.format(sub, rel, obj))
                    continue

                id = raw_data['_id']
                origin_triple = raw_data['triple']
                freebase_link = raw_data['true_link']
                wiki_link = raw_data['entity_linking']
                src_sentences = raw_data['src_sentences']

                side_info = SideInfo(origin_triple, freebase_link, wiki_link, src_sentences)
                mention = Mention(sub, rel, obj, id, side_info)

                self.mentions_list.append(mention)
        self.fix_typo()
        self.get_ground_truth()
