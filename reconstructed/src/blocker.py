from collections import defaultdict as ddict

import numpy as np

'''
Blocker takes mentions_list as input
and return a list of list of mentions ie. [[mention1, .. ], [mentionX, ..]]
'''
class Blocker(object):

    def __init__(self, args):
        self.args = args
        self.result = []

    '''
    This method block by strict string comparison
    '''
    def strict_block(self, mentions_list):
        self.result = []
        sub2mentions = ddict(list)
        for mention in mentions_list:
            sub2mentions[mention.sub].append(mention)

        for _, mention_cluster in sub2mentions.items():
            self.result.append(mention_cluster)

        return self.result, self.result

    '''
    O(n^2) implementation, very slow for large mentions_list
    '''
    def hierarchy_block_1(self, mentions_list):
        self.result = []
        sorted_mentions_list = sorted(mentions_list, key = lambda x: len(x.sub), reverse=True)
        length = len(sorted_mentions_list)
        flag = np.zeros(length)
        sub2mentions = ddict(list)
        for index, mention in enumerate(sorted_mentions_list):
            if flag[index] == 1: continue
            sub2mentions[mention.sub].append(mention)
            flag[index] = 1
            for i in range(index+1, length):
                if flag[i] == 1: continue
                if sorted_mentions_list[i].sub in mention.sub:
                    sub2mentions[mention.sub].append(sorted_mentions_list[i])
                    flag[i] = 1

        count = 0
        for _, mention_cluster in sub2mentions.items():
            count += len(mention_cluster)
            self.result.append(mention_cluster)

        assert(count == length)
        return self.result, self.result

    def hierarchy_block_2(self, mentions_list):
        self.result = []
        sorted_mentions_list = sorted(mentions_list, key = lambda x: len(x.sub), reverse=True)
        length = len(sorted_mentions_list)
        flag = np.zeros(length)
        sub2mentions = ddict(list)
        for index, mention in enumerate(sorted_mentions_list):
            sub2mentions[mention.sub].append(mention)
            for i in range(index+1, length):
                if sorted_mentions_list[i].sub in mention.sub:
                    sub2mentions[mention.sub].append(sorted_mentions_list[i])

        for _, mention_cluster in sub2mentions.items():
            self.result.append(mention_cluster)

        result_c = []
        sub2mentions_c = ddict(list)
        for mention in mentions_list:
            sub2mentions_c[mention.sub].append(mention)

        for _, mention_cluster in sub2mentions_c.items():
            result_c.append(mention_cluster)
        return self.result, result_c


    def idf_block(self):
        pass
