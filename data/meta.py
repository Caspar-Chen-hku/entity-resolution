import json
from collections import defaultdict as ddict

import codecs


def print_meta(triples_list):
    num_triples = len(triples_list)

    label2trp = ddict(int)
    for trp in triples_list:
        label2trp[trp['label']] += 1

    num_clusters = 0
    largest_cluster = -1
    singletons = 0
    for k, v in label2trp.items():
        num_clusters += 1
        if v == 1: singletons += 1
        if v > largest_cluster:
            largest_cluster = v

    print('number of triples: {}'.format(num_triples))
    print('number of clusters: {}'.format(num_clusters))
    print('number of singletons: {}'.format(singletons))
    print('largest cluster: {}'.format(largest_cluster))
    print('========================')


triples_list = []
with open('ambiguous.tsv') as f:
    for line in f:
        data = line.split('\t')
        if data[0] == '==CLUSTER==\n': continue
        sub,rel,obj = str(data[1]), str(data[4]), str(data[5])
        trp = {}
        trp['id'] = int(data[3])
        trp['triple'] = [sub, rel, obj]
        trp['label'] = str(data[0])
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        trp['label'] = trp['true_link']['subject']
        triples_list.append(trp)
    print_meta(triples_list)


triples_list = []
with codecs.open('reverb45k', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        trp['label'] = trp['entity_linking']['subject']
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k_wiki_only', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        trp['label'] = trp['entity_linking']['subject']
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k_intersect', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k_intersect2', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k_union', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        triples_list.append(trp)
    print_meta(triples_list)

triples_list = []
with codecs.open('reverb45k_union2', encoding='utf-8', errors='ignore') as f:
    for line in f:
        trp = json.loads(line.strip())
        triples_list.append(trp)
    print_meta(triples_list)
