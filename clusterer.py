import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from helper import *

class Clusterer():

    def __init__(self, args, glove_embedding, sub_list, sub2trp, triples_list):
        self.p = args
        self.glove_embedding = glove_embedding
        self.sub_list = sub_list
        self.sub2trp = sub2trp
        self.triples_list = triples_list

    def get_level1_cluster(self):
        dist = pdist(self.glove_embedding, metric=self.p.metric)
        clust_res = linkage(dist, method=self.p.linkage)
        labels = fcluster(clust_res, t=self.p.thresh_val, criterion='distance') -1

        self.level1_clusters = [[] for i in range(max(labels) + 1)]
        for i in range(len(labels)):
            self.level1_clusters[labels[i]].append(i)

    def interpret_level1_cluster(self):
        self.x_ent2clust_u = ddict(set)
        for i in range(len(self.level1_clusters)):
            cluster = self.level1_clusters[i]
            for sub_id in cluster:
                for trp in self.sub2trp[self.sub_list[sub_id]]:
                    if trp['no_shared_word_within_entity'] is 0: continue
                    if trp['ent_lnk_sub'] is None: continue
                    sub_u = trp['triple_unique'][0]
                    self.x_ent2clust_u[sub_u].add(i)
        self.x_clust2ent_u = invert_dict(self.x_ent2clust_u, 'm2os')

    def get_level2_cluster_bert(self):
        self.level2_clusters = []
        for i in range(len(self.level1_clusters)):
            cluster = self.level1_clusters[i]
            bert_embed_list = []
            tmp_trp = []
            for sub_id in cluster:
                for trp in self.sub2trp[self.sub_list[sub_id]]:
                    tmp_trp.append(trp)
                    bert_embed_list.append(trp['sub_bert_vec'])
            bert_embed_list = np.array(bert_embed_list)

            if len(bert_embed_list) == 1:
                self.level2_clusters.append(tmp_trp)
                continue

            dist = pdist(bert_embed_list, metric=self.p.metric)
            clust_res = linkage(dist, method=self.p.linkage)
            labels = fcluster(clust_res, t=self.p.b_thresh_val, criterion='distance') - 1

            tmp_clusters = [[] for i in range(max(labels) + 1)]
            for i in range(len(labels)):
                tmp_clusters[labels[i]].append(tmp_trp[i])

            for tmp_cluster in tmp_clusters:
                tmp = []
                for trp in tmp_cluster:
                    tmp.append(trp)
                self.level2_clusters.append(tmp)

        self.x_ent2clust_u = ddict(set)
        for i in range(len(self.level2_clusters)):
            cluster = self.level2_clusters[i]
            for trp in cluster:
                if trp['share_word_with_other_entity'] is 0: continue
                if trp['ent_lnk_sub'] is None: continue
                sub_u = trp['triple_unique'][0]
                self.x_ent2clust_u[sub_u].add(i)
        self.x_clust2ent_u = invert_dict(self.x_ent2clust_u, 'm2os')


    def get_level2_cluster(self):
        self.level2_clusters = []
        for i in range(len(self.level1_clusters)):
            cluster = self.level1_clusters[i]
            embed_list = []
            for sub_id in cluster:
                vec = np.zeros(768)
                for trp in self.sub2trp[self.sub_list[sub_id]]:
                    vec += trp['sub_bert_vec']
                embed_list.append(vec / len(cluster))
            embed_list = np.array(embed_list)

            if len(embed_list) == 1:
                self.level2_clusters.append(cluster)
                continue

            dist = pdist(embed_list, metric=self.p.metric)
            clust_res = linkage(dist, method=self.p.linkage)
            labels = fcluster(clust_res, t=self.p.b_thresh_val, criterion='distance') -1

            tmp_clusters = [[] for i in range(max(labels) + 1)]
            for i in range(len(labels)):
                tmp_clusters[labels[i]].append(i)

            for tmp_cluster in tmp_clusters:
                tmp = []
                for tmp_id in tmp_cluster:
                    sub_id = cluster[tmp_id]
                    tmp.append(sub_id)
                self.level2_clusters.append(tmp)


    def interpret_level2_cluster(self):
        self.x_ent2clust_u = ddict(set)
        for i in range(len(self.level2_clusters)):
            cluster = self.level2_clusters[i]
            for sub_id in cluster:
                for trp in self.sub2trp[self.sub_list[sub_id]]:
                    if trp['no_shared_word_within_entity'] is 0: continue
                    if trp['ent_lnk_sub'] is None: continue
                    sub_u = trp['triple_unique'][0]
                    self.x_ent2clust_u[sub_u].add(i)
        self.x_clust2ent_u = invert_dict(self.x_ent2clust_u, 'm2os')

    def bert_cluster(self):
        bert_embed_list = []
        for trp in self.triples_list:
            bert_embed_list.append(trp['sub_bert_vec'])
        bert_embed_list = np.array(bert_embed_list)

        dist = pdist(bert_embed_list, metric=self.p.metric)
        clust_res = linkage(dist, method=self.p.linkage)
        labels = fcluster(clust_res, t=self.p.b_thresh_val, criterion='distance') -1
        bert_clusters = [[] for i in range(max(labels)+1)]
        for i in range(len(labels)):
            bert_clusters[labels[i]].append(i)

        self.x_ent2clust_u = ddict(set)
        for i in range(len(bert_clusters)):
            cluster = bert_clusters[i]
            for trp_id in cluster:
                trp = self.triples_list[trp_id]
                if trp['ent_lnk_sub'] is None: continue
                sub_u = trp['triple_unique'][0]
                self.x_ent2clust_u[sub_u].add(i)
        self.x_clust2ent_u = invert_dict(self.x_ent2clust_u, 'm2os')

    def get_return(self):
        return self.x_ent2clust_u, self.x_clust2ent_u
