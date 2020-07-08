from helper import *
from gensim.utils import lemmatize

import csv

class Reader():

    def __init__(self, args, logger):
        self.p = args
        self.logger = logger
        #e.g. data/reverb45k
        self.data_file = 'data/' + self.p.dataset
        #e.g. output/running_name/reverb45k_triples.txt
        self.cache_file = self.p.out_path + '/' + self.p.dataset + '_triples.txt'
        self.ground_truth_file = self.p.out_path + '/ground_truth.txt'
        self.triples_list = []
        self.is_acronym = {}
        self.amb_ent = ddict(int)
        self.amb_mentions = {}

    def read_reverb45k(self):
        if not check_file(self.cache_file):
            count = 0
            self.logger.info('\t Read from raw reverb45k file')
            with codecs.open(self.data_file, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    tmp = json.loads(line.strip())

                    sub, rel, obj = map(str, tmp['triple_norm'])

                    if sub.isalpha() and sub.isupper(): self.is_acronym[sub] = 1
                    if obj.isalpha() and obj.isupper(): self.is_acronym[obj] = 1

                    sub = self._normalize_ent(sub)
                    obj = self._normalize_ent(obj)

                    if len(sub) == 0 or len(rel) == 0 or len(obj) == 0:
                        continue

                    trp = {}
                    if int(tmp['_id']) == 114844:
                        count += 1
                        continue #bug fixing, somehow bert encode lost this triple

                    trp['id'] = tmp['_id']
                    trp['triple_raw'] = tmp['triple']
                    trp['triple'] = [sub, rel, obj]
                    trp['triple_unique'] = [sub+'|'+str(tmp['_id']), rel, obj+'|'+str(tmp['_id'])]
                    trp['ent_lnk_sub'] = tmp['entity_linking']['subject']
                    trp['ent_lnk_obj'] = tmp['entity_linking']['object']
                    trp['true_sub_link'] = tmp['true_link']['subject']
                    trp['true_obj_link'] = tmp['true_link']['object']
                    trp['kbp_info'] = tmp['kbp_info']
                    trp['src_sentences'] = tmp['src_sentences']
                    trp['no_shared_word_within_entity'] = None
                    trp['share_word_with_other_entity'] = None
                    if 'label' in tmp:
                        trp['label'] = tmp['label']
                    else:
                        trp['label'] = trp['ent_lnk_sub']
                    if 'no_shared_word_within_entity' in tmp:
                        trp['no_shared_word_within_entity'] = tmp['no_shared_word_within_entity']
                    if 'share_word_with_other_entity' in tmp:
                        trp['share_word_with_other_entity'] = tmp['share_word_with_other_entity']

                    self.triples_list.append(trp)
                print(count)
                with open(self.cache_file, 'w') as f:
                    f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
        else:
            self.logger.info('\tLoading cached reverb45k triples')
            with open(self.cache_file) as f:
                self.triples_list = [json.loads(triple) for triple in f.read().split('\n')]
        self.logger.info('\t Number of triples: {}'.format(len(self.triples_list)))
        assert(1==0)
        ''' ambiguous entities '''
        self.amb_clust = {}
        for trp in self.triples_list:
            sub = trp['triple'][0]
            for tok in sub.split():
                self.amb_clust[tok] = self.amb_clust.get(tok, set())
                self.amb_clust[tok].add(sub)

        for rep, clust in self.amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1

        ''' ground truth '''
        self.true_ent2clust = ddict(set)
        for trp in self.triples_list:
             if trp['ent_lnk_sub'] is None: continue
             if trp['no_shared_word_within_entity'] is 0: continue
             sub_u = trp['triple_unique'][0]
             self.true_ent2clust[sub_u].add(trp['label'])
        self.true_clust2ent = invert_dict(self.true_ent2clust, 'm2os')

        if not check_file(self.ground_truth_file):
            self.logger.info('\t Printing ground truth')
            self._print_ground_truth()
        else: self.logger.info('\t Grount truth exist')

    def read_ambiguous(self):
        fname = 'data/ambiguous.tsv'
        with open(fname) as f:
            for line in f:
                data = line.split('\t')
                if data[0] == '==CLUSTER==\n':  continue
                sub,rel,obj = str(data[1]), str(data[4]), str(data[5])

                if sub.isalpha() and sub.isupper(): self.is_acronym[sub] = 1
                if obj.isalpha() and obj.isupper(): self.is_acronym[obj] = 1

                sub = self._normalize_ent(sub)
                obj = self._normalize_ent(obj)

                if len(sub) == 0 or len(rel) == 0 or len(obj) == 0: continue

                trp = {}
                trp['id'] = int(data[3])
                trp['triple'] = [sub, rel, obj]
                trp['triple_fixed'] = [sub, rel, obj] # no fix needed for ambiguous data
                trp['triple_unique'] = [sub+'|'+str(trp['id']), rel, obj+'|'+str(trp['id'])]
                trp['true_sub_link'] = str(data[0])
                trp['ent_link_sub'] = None
                trp['label'] = trp['true_sub_link']

                self.triples_list.append(trp)
        self.logger.info('\t Number of triples: {}'.format(len(self.triples_list)))

        '''ground truth'''
        self.true_ent2clust = ddict(set)
        for trp in self.triples_list:
             sub_u = trp['triple_unique'][0]
             self.true_ent2clust[sub_u].add(trp['true_sub_link'])
        self.true_clust2ent = invert_dict(self.true_ent2clust, 'm2os')

        if not check_file(self.ground_truth_file):
            self.logger.info('\t Printing ground truth')
            self._print_ground_truth()
        else: self.logger.info('\t Grount truth exist')

    def read_ambiguous_cesi(self):
        self.logger.info('\t Read from cesi ambiguous file')
        with open('data/ambiguous/ambiguous_test') as f:
            for line in f:
                tmp = json.loads(line.strip())

                sub, rel, obj = map(str, tmp['triple_norm'])

                if sub.isalpha() and sub.isupper(): self.is_acronym[sub] = 1
                if obj.isalpha() and obj.isupper(): self.is_acronym[obj] = 1

                sub = self._normalize_ent(sub)
                obj = self._normalize_ent(obj)

                if len(sub) == 0 or len(rel) == 0 or len(obj) == 0: continue

                trp = {}
                trp['id'] = tmp['_id']
                trp['triple_raw'] = tmp['triple']
                trp['triple'] = [sub, rel, obj]
                trp['triple_unique'] = [sub+'|'+str(tmp['_id']), rel, obj+'|'+str(tmp['_id'])]
                trp['ent_lnk_sub'] = tmp['entity_linking']['subject']
                trp['ent_lnk_obj'] = tmp['entity_linking']['object']
                trp['true_sub_link'] = tmp['true_link']['subject']
                trp['true_obj_link'] = tmp['true_link']['object']
                trp['kbp_info'] = tmp['kbp_info']
                trp['src_sentences'] = tmp['src_sentences']
                trp['label'] = trp['ent_lnk_sub']

                self.triples_list.append(trp)

        ''' ambiguous entities '''
        self.amb_clust = {}
        for trp in self.triples_list:
            sub = trp['triple'][0]
            for tok in sub.split():
                self.amb_clust[tok] = self.amb_clust.get(tok, set())
                self.amb_clust[tok].add(sub)

        for rep, clust in self.amb_clust.items():
            if rep in clust and len(clust) >= 3:
                self.amb_ent[rep] = len(clust)
                for ele in clust: self.amb_mentions[ele] = 1

        ''' ground truth '''
        self.true_ent2clust = ddict(set)
        for trp in self.triples_list:
             if trp['ent_lnk_sub'] is None: continue
             sub_u = trp['triple_unique'][0]
             self.true_ent2clust[sub_u].add(trp['label'])
        self.true_clust2ent = invert_dict(self.true_ent2clust, 'm2os')

        if not check_file(self.ground_truth_file):
            self.logger.info('\t Printing ground truth')
            self._print_ground_truth()
        else: self.logger.info('\t Grount truth exist')

    def read_base(self):
        pass

    def _normalize_ent(self, ent):
        ent = ent.lower().replace('.', ' ').replace('-', ' ').strip().replace('_',' ').replace('|', ' ').strip()
        ent = ' '.join([ tok.decode('utf-8').split('/')[0] for tok in lemmatize(ent)])
        return ent

    def _print_ground_truth(self):
        id2trp = {}
        for trp in self.triples_list:
            id2trp[trp['id']] = trp
        with open(self.ground_truth_file, 'w') as f:
            for k, v in self.true_clust2ent.items():
                f.write('{}: \n'.format(k))
                for ent in v:
                    name, id = ent.split('|')
                    [sub, rel, obj] = id2trp[int(id)]['triple']
                    src = sub + ' ' + rel + ' ' + obj
                    f.write('id: {}, sub: {}, src: {}\n'.format(id, name, src))
        '''
        with open(self.p.out_path + '/amb_clust.txt', 'w') as f2:
            for k, v in self.amb_clust.items():
                f2.write('{}: {}\n'.format(k, v))
        '''

    def get_return(self):
        return self.triples_list, self.is_acronym, self.amb_ent, self.amb_mentions, self.true_ent2clust, self.true_clust2ent
