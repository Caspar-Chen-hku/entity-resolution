import argparse
import os
from collections import defaultdict as ddict

from reader import AmbiguousReader, Reverb45kReader
from blocker import Blocker
from clusterer import Cluster
from evaluater import evaluate
from embedding.embeder import GloveEmbeder, BertEmbeder
from utils.helper import *
from utils.printer import *


class MLCE_Main(object):

    def __init__(self, args):
        self.args = args
        self.logger = get_logger(self.args.name)
        self.logger.info('Running {}'.format(self.args.name))

    def read_data(self):
        self.logger.info('Reading dataset {}_{}'.format(self.args.dataset, self.args.split))
        #select dataset
        if self.args.dataset == 'reverb45k':
            reader = Reverb45kReader(self.args)
        elif self.args.dataset == 'ambiguous':
            reader = AmbiguousReader(self.args)
        else:
            print('Unkown Dataset')
            assert(1==0)

        reader.process()
        self.mentions_list, self.true_ent2clust, self.true_clust2ent = reader.get_return()
        #self.mentions_list = self.mentions_list[0:1000]

    def process(self):
        #init
        self.logger.info('Initializing classes')
        blocker = Blocker(self.args)
        g_embeder = GloveEmbeder(self.args)
        b_embeder = BertEmbeder(self.args)
        b_embeder.load_bert_encode()
        cluster = Cluster(self.args)

        # step 1: Glove
        self.logger.info('Level 1 Glove embedding')
        blocked_canopies, _ = blocker.strict_block(self.mentions_list)
        embeddings = g_embeder.embed(blocked_canopies)
        self.result = cluster.process(blocked_canopies, embeddings, self.args.thresh_val)

        #step 2: Bert
        if self.args.mode == 'mlce':
            self.logger.info('Level 2 Bert embedding')
            canopies = self.result
            tmp = []
            for canopy in canopies:
                blocked_canopies, blocked_canopies_c = blocker.strict_block(canopy)
                #blocked_canopies, blocked_canopies_c = blocker.hierarchy_block_2(canopy)
                b_embeddings = b_embeder.embed(blocked_canopies)
                if len(blocked_canopies_c) == 1:
                    tmp += blocked_canopies_c
                else:
                    tmp += cluster.process(blocked_canopies_c, b_embeddings, self.args.b_thresh_val)
            self.result = tmp

    def evaluate(self):
        self.x_ent2clust_u = ddict(set)
        for index, cluster in enumerate(self.result):
            for mention in cluster:
                if mention.omit: continue
                self.x_ent2clust_u[mention.sub_u].add(index)
        self.x_clust2ent_u = invert_dict(self.x_ent2clust_u, 'm2os')
        print_result(self.args.label, self.mentions_list, self.x_clust2ent_u, self.args.out_path + '/result')

        self.logger.info('Evaluating')
        eval_results = evaluate(self.x_ent2clust_u, self.x_clust2ent_u, self.true_ent2clust, self.true_clust2ent)

        self.logger.info('Macro Precision: {}, Macro Recall: {}, Macro F1: {}\n'.format(eval_results['macro_prec'], eval_results['macro_recall'], eval_results['macro_f1']))
        self.logger.info('Micro Precision: {}, Micro Recall: {}, Micro F1: {}\n'.format(eval_results['micro_prec'], eval_results['micro_recall'], eval_results['micro_f1']))
        self.logger.info('Pair Precision: {}, Pair Recall: {}, Pair F1: {}\n'.format(eval_results['pair_prec'], eval_results['pair_recall'], eval_results['pair_f1']))
        self.logger.info('CESI: #Clusters: %d, #Singletons %d\n'    % (len(self.x_clust2ent_u), 	len([1 for _, clust in self.x_clust2ent_u.items()    if len(clust) == 1])))
        self.logger.info('Gold: #Clusters: %d, #Singletons %d\n' % (len(self.true_clust2ent),  len([1 for _, clust in self.true_clust2ent.items() if len(clust) == 1])))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # experiment arguments
    parser.add_argument('-name', dest='name', default='test', help='Running name')
    parser.add_argument('-dataset', dest='dataset', default='reverb45k', help='Dataset used for experiment')
    parser.add_argument('-split', dest='split', default='test', help='test or valid set')
    parser.add_argument('-mode', dest='mode', default='mlce', help='choose among [mlce, glove]')
    parser.add_argument('-label', dest='label', default='wiki', help='wiki or freebase as ground truth')
    parser.add_argument('-omit', dest='omit', action='store_true', help='omit None ground truth')
    parser.add_argument('-reset', dest='reset', action='store_true', help='Clear cached files')

    parser.add_argument('-metric', dest='metric', default='cosine', help='Metric for calculating distance between embeddings')
    parser.add_argument('-linkage', dest='linkage', default='complete', choices=['complete', 'single', 'avergage'], help='HAC linkage criterion')
    parser.add_argument('-thresh_val', dest='thresh_val', default=0.36, type=float, help='Threshold for clustering')
    parser.add_argument('-b_thresh_val', dest='b_thresh_val', default=0.33, type=float, help='Threshold for bert clustering')

    args = parser.parse_args()
    args.out_path = '../output/' + args.name

    # clear catched files
    if args.reset:
        os.system('rm -r {}'.format(args.out_path))
        os.system('rm log/{}'.format(args.name))

    # create output directory
    if not os.path.isdir(args.out_path):
        os.system('mkdir -p {}'.format(args.out_path))

    mlce = MLCE_Main(args)
    mlce.read_data()
    mlce.process()
    mlce.evaluate()
