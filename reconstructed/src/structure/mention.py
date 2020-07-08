from gensim.utils import lemmatize

# Data structure for storing mentions
class Mention(object):

    def __init__(self, sub, rel, obj, id, side_info):
        self.id = id
        self.sub = self._normalize(sub)
        self.sub_u = '{}|{}'.format(self.sub, str(self.id))
        self.rel = self._normalize(rel)
        self.obj = self._normalize(obj)
        self.obj_u = '{}|{}'.format(self.obj, str(self.id))
        self.side_info = side_info
        self.label = None
        self.omit = None # whether omit in evaluation
    
    def __str__(self):
        return '{}| {} {} {}'.format(self.id, self.sub, self.rel, self.obj)
    
    def __eq__(self, other):
        return self.label == other.label
    
    def _normalize(self, ent):
        ent = ent.lower().replace('.', ' ').replace('-', ' ').strip()
        ent = ent.replace('_', ' ').replace('|', ' ').strip()
        ent = ' '.join([tok.decode('utf-8').split('/')[0] for tok in lemmatize(ent)])
        return ent