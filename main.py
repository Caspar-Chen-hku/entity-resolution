from helper import *
from reader import Reader
from cleaner import Cleaner
from embeder import Embeder
from clusterer import Clusterer
from evaluater import evaluate

class CESI_Main():

    def __init__(self, args):
        self.p = args
        self.logger = get_logger(self.p.name)
        self.logger.info('Running {}'.format(self.p.name))
        self.logger.info('Parameters:')
        self.logger.info('\t dataset: {}'.format(self.p.dataset))
        self.logger.info('\t level 1 threshold: {}'.format(self.p.thresh_val))
        self.logger.info('\t level 2 threshold: {}'.format(self.p.b_thresh_val))

    def read_triples(self):
        self.logger.info('Reading dataset {}'.format(self.p.dataset))

        reader = Reader(self.p, self.logger)
        if 'reverb45k' in self.p.dataset:
            reader.read_reverb45k()
        elif self.p.dataset == 'ambiguous':
            reader.read_ambiguous_cesi()
        elif self.p.dataset == 'base':
            reader.read_base()
        else:
            print('Dataset undefined!')
            assert(1==0)
        self.triples_list, self.is_acronym, self.amb_ent, self.amb_mentions, self.true_ent2clust, self.true_clust2ent = reader.get_return()

    def clean_triples(self):
        self.logger.info('Cleaning data')

        cleaner = Cleaner(self.p, self.triples_list, self.amb_ent, self.is_acronym, self.true_clust2ent)
        self.sub_list, self.sub2trp, self.triples_list = cleaner.get_return()

    def embed_triples(self):
        self.logger.info('Embedding triples')

        embeder = Embeder(self.sub_list, self.sub2trp, self.triples_list)
        self.glove_embedding, self.triples_list =  embeder.get_return()

    def cluster(self):
        self.logger.info('Clustering triples')

        clusterer = Clusterer(self.p, self.glove_embedding, self.sub_list, self.sub2trp, self.triples_list)
        clusterer.get_level1_cluster()
        #clusterer.interpret_level1_cluster()
        clusterer.get_level2_cluster()
        clusterer.interpret_level2_cluster()
        #clusterer.bert_cluster()

        self.x_ent2clust_u, self.x_clust2ent_u = clusterer.get_return()

    def evaluate(self):
        self.logger.info('Evaluating')

        eval_results = evaluate(self.x_ent2clust_u, self.x_clust2ent_u, self.true_ent2clust, self.true_clust2ent)

        self.logger.info('Macro Precision: {}, Macro Recall: {}, Macro F1: {}\n'.format(eval_results['macro_prec'], eval_results['macro_recall'], eval_results['macro_f1']))
        self.logger.info('Micro Precision: {}, Micro Recall: {}, Micro F1: {}\n'.format(eval_results['micro_prec'], eval_results['micro_recall'], eval_results['micro_f1']))
        self.logger.info('Pair Precision: {}, Pair Recall: {}, Pair F1: {}\n'.format(eval_results['pair_prec'], eval_results['pair_recall'], eval_results['pair_f1']))
        self.logger.info('CESI: #Clusters: %d, #Singletons %d\n'    % (len(self.x_clust2ent_u), 	len([1 for _, clust in self.x_clust2ent_u.items()    if len(clust) == 1])))
        self.logger.info('Gold: #Clusters: %d, #Singletons %d\n' % (len(self.true_clust2ent),  len([1 for _, clust in self.true_clust2ent.items() if len(clust) == 1])))

        '''print result'''
        id2trp = {}
        for trp in self.triples_list:
            id2trp[trp['id']] = trp
        fn = self.p.out_path + '/cluster_result.txt'
        with open(fn, 'w') as f:
            for k, v in self.x_clust2ent_u.items():
                f.write('{}: \n'.format(k))
                for ent in v:
                    name, id = ent.split('|')
                    [sub, rel, obj] = id2trp[int(id)]['triple']
                    src = sub + ' ' + rel + ' ' + obj
                    f.write('id: {}, sub: {}, src: {}\n'.format(id, name, src))

if __name__ == '__main__':
    #parse the arguments
    parser = argparse.ArgumentParser()
    #arguments expected to be freezed

    #arguments expected to be changed for each run
    parser.add_argument('-name', dest='name', default='name', help='Running name')
    parser.add_argument('-dataset', dest='dataset', default='reverb45k', help='Dataset used for experiment')
    parser.add_argument('-reset', dest='reset', action='store_true', help='Clear cached files')

    parser.add_argument('-linkage', 	dest='linkage',    default='complete', choices=['complete', 'single', 'avergage'], help='HAC linkage criterion')
    parser.add_argument('-thresh_val', 	dest='thresh_val', 	default=.36, 		type=float, 	help='Threshold for clustering')
    parser.add_argument('-metric', 		dest='metric', 		default='cosine', 			help='Metric for calculating distance between embeddings')
    parser.add_argument('-b_thresh_val', 	dest='b_thresh_val', 	default=.33, 		type=float, 	help='Threshold for bert clustering')
    args = parser.parse_args()
    args.out_path = 'output/' + args.name
    if args.reset:
        os.system('rm -r {}'.format(args.out_path))
        os.system('rm log/{}'.format(args.name))
    if not os.path.isdir(args.out_path):
        os.system('mkdir -p ' + args.out_path)

    #start process
    cesi = CESI_Main(args)
    cesi.read_triples()
    cesi.clean_triples()
    cesi.embed_triples()
    cesi.cluster()
    cesi.evaluate()
    cesi.logger.info('End\n\n')
