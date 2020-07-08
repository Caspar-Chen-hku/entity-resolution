import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

'''
Cluster takes embedding as input
and return canopies ie. [[mentions1, ..], [mentionsX, ..]]
'''
class Cluster(object):

    def __init__(self, args):
        self.args = args
        self.result = []

    def process(self, canopies, embeddings, thresh_val):
        dist = pdist(embeddings, metric=self.args.metric)
        clust_res = linkage(dist, method=self.args.linkage)
        labels = fcluster(clust_res, t=thresh_val, criterion='distance') - 1

        self.result = [[] for i in range(max(labels)+1)]
        for i in range(len(labels)):
            self.result[labels[i]] += canopies[i]
        return self.result
