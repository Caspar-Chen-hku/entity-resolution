from collections import defaultdict as ddict

'''
print ground truth in following format
===Cluster===
subject, the other ground truth, source sentence
'''
def print_ground_truth(label, mentions_list, true_clust2ent, path):
    id2mention = {}
    for mention in mentions_list:
        id2mention[mention.id] = mention

    if label == 'freebase':
        with open(path, 'w') as f:
            for _, v in true_clust2ent.items():
                f.write('==Cluster==\n')
                for ent in v:
                    _, id = ent.split('|')
                    mention = id2mention[int(id)]
                    f.write('{} || {} || {}\n'.format(mention.sub, mention.side_info.wiki_link_sub, mention.side_info.src_sentences[0]))
    elif label == 'wiki':
        with open(path, 'w') as f:
            for _, v in true_clust2ent.items():
                f.write('==Cluster==\n')
                for ent in v:
                    _, id = ent.split('|')
                    mention = id2mention[int(id)]
                    f.write('{} || {} || {}\n'.format(mention.sub, mention.side_info.freebase_link_sub, mention.side_info.src_sentences[0]))

'''
This function print the difference between wiki and freebase as ground truth
'''
def print_ground_truth_diff(label, mentions_list, true_clust2ent, path, rate):
    id2mention = {}
    for mention in mentions_list:
        id2mention[mention.id] = mention

    if label == 'freebase':
        with open(path + '_' + str(rate), 'w') as f:
            for _, v in true_clust2ent.items():
                total_count = 0
                wiki_count = ddict(int)
                for ent in v:
                    _, id = ent.split('|')
                    mention = id2mention[int(id)]
                    if mention.side_info.wiki_link_sub is None: continue
                    total_count += 1
                    wiki_count[mention.side_info.wiki_link_sub] += 1

                if total_count == 0 or not wiki_count: continue

                max_wiki_count = wiki_count[max(wiki_count, key=wiki_count.get)]
                if max_wiki_count / total_count < rate:
                    f.write('==Cluster==\n')
                    for ent in v:
                        _, id = ent.split('|')
                        mention = id2mention[int(id)]
                        f.write('{} || {} || {}\n'.format(mention.sub, mention.side_info.wiki_link_sub, mention.side_info.src_sentences[0]))

def print_result(label, mentions_list, result_clust2ent, path):
    id2mention = {}
    for mention in mentions_list:
        id2mention[mention.id] = mention

    if label == 'freebase':
        with open(path, 'w') as f:
            for _, v in result_clust2ent.items():
                f.write('==Cluster==\n')
                for ent in v:
                    _, id = ent.split('|')
                    mention = id2mention[int(id)]
                    f.write('{} || {} || {}\n'.format(mention.sub, mention.side_info.freebase_link_sub, mention.side_info.src_sentences[0]))
    elif label == 'wiki':
        with open(path, 'w') as f:
            for _, v in result_clust2ent.items():
                f.write('==Cluster==\n')
                for ent in v:
                    _, id = ent.split('|')
                    mention = id2mention[int(id)]
                    f.write('{} || {} || {}\n'.format(mention.sub, mention.side_info.wiki_link_sub, mention.side_info.src_sentences[0]))
