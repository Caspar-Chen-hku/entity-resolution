class SideInfo(object):

    def __init__(self, origin_triple, freebase_link, wiki_link, src_sentences):
        self.origin_triple = origin_triple
        self.freebase_link_sub = freebase_link['subject']
        self.freebase_link_obj = freebase_link['object']
        self.wiki_link_sub = wiki_link['subject']
        self.wiki_link_obj = wiki_link['object']
        self.src_sentences = src_sentences