import codecs, json

class Calibrator():

    def __init__(self):
        self.raw_triples = []
        self.calibrated_triples = []
        self.read_data()

    def read_data(self):
        with codecs.open('reverb45k_wiki_only', encoding='utf-8', errors='ignore') as f:
            for line in f:
                trp = json.loads(line.strip())
                self.raw_triples.append(trp)

    def wiki_only(self):
        for trp in self.raw_triples:
            if trp['entity_linking']['subject'] is None:
                continue
            else:
                self.calibrated_triples.append(trp)
        print(len(self.calibrated_triples))
        with open('reverb45k_cali_wiki_only', 'w') as f:
            f.write('\n'.join([json.dumps(triple) for triple in self.calibrated_triples]))

    def check(self):
        no_true_link = 0
        no_wiki_link = 0
        for trp in self.raw_triples:
            if trp['entity_linking']['subject'] is None:
                no_wiki_link += 1
            if trp['true_link']['subject'] is None:
                no_true_link += 1
        print('no true link: {}'.format(no_true_link))
        print('no wiki link: {}'.format(no_wiki_link))

    def true_plus_wiki(self):
        link_dict = {}
        mention_dict = {}
        new_mention_dict_index = 0

        for trp in self.raw_triples:
            subject = trp['triple_norm'][0]
            fbid = trp['true_link']['subject']
            if fbid is None: continue
            wikilink = trp['entity_linking']['subject']
            if wikilink is None: continue

            if wikilink is None:
                if link_dict.get(fbid) is None:
                    mention_dict[new_mention_dict_index] = [trp]
                    link_dict[fbid] = new_mention_dict_index
                    new_mention_dict_index += 1
                else:
                    mention_dict[link_dict.get(fbid)].append(trp)
            else:
                if link_dict.get(fbid) is None and link_dict.get(wikilink) is None:
                    mention_dict[new_mention_dict_index] = [trp]
                    link_dict[fbid] = new_mention_dict_index
                    link_dict[wikilink] = new_mention_dict_index
                    new_mention_dict_index += 1
                elif link_dict.get(fbid) is None and link_dict.get(wikilink) is not None:
                    mention_dict[link_dict.get(wikilink)].append(trp)
                    link_dict[fbid] = link_dict.get(wikilink)
                elif link_dict.get(fbid) is not None and link_dict.get(wikilink) is None:
                    mention_dict[link_dict.get(fbid)].append(trp)
                    link_dict[wikilink] = link_dict.get(fbid)
                elif link_dict.get(fbid) is not None and link_dict.get(wikilink) is not None:
                    if link_dict.get(fbid) == link_dict.get(wikilink):
                        mention_dict[link_dict.get(fbid)].append(trp)
                    else:
                        mention_dict[link_dict.get(fbid)].append(trp)
                        merged_key = link_dict.get(wikilink)

                        mention_dict[link_dict.get(fbid)].extend(mention_dict.pop(merged_key))
                        for k, v in link_dict.items():
                            if v == merged_key:
                                link_dict[k] = link_dict[fbid]

        count = 0
        for k, v in mention_dict.items():
            count += 1
            for ent in v:
                ent['label'] = k
                self.calibrated_triples.append(ent)
        with open('reverb45k_union2', 'w') as f:
            f.write('\n'.join([json.dumps(triple) for triple in self.calibrated_triples]))

    def true_minus_wiki(self):
        # initial
        label = 0
        for trp in self.raw_triples:
            trp['label'] = label

        label += 1
        for trp1 in self.raw_triples:
            if trp1['label'] != 0: continue
            trp1['label'] = label
            for trp2 in self.raw_triples:
                if trp1['_id'] == trp2['_id']: continue
                if trp2['label'] != 0: continue
                # no wiki link than only look and true link
                if trp1['entity_linking']['subject'] is None or trp2['entity_linking']['subject'] is None:
                    if trp1['true_link']['subject'] == trp2['true_link']['subject']:
                        trp2['label'] = label
                else:
                    if trp1['entity_linking']['subject'] == trp2['entity_linking']['subject'] \
                        and trp1['true_link']['subject'] == trp2['true_link']['subject']:
                        trp2['label'] = label
            label += 1
        self.calibrated_triples = self.raw_triples
        #print(label)
        with open('reverb45k_intersect2', 'w') as f:
            f.write('\n'.join([json.dumps(triple) for triple in self.calibrated_triples]))


calibrator = Calibrator()
#calibrator.wiki_only()
#calibrator.check()
calibrator.true_plus_wiki()
calibrator.true_minus_wiki()
