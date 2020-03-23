from corpus import Corpus

from os import path as op
import pickle




class Candidate:
    def _candidates_of(self, mentions, seq):
        raise NotImplemented("从序列和相关提及中获取候选")

    def add_candidates_to_corpus(self, corpus: Corpus):
        raise NotImplemented("添加候选到语料库中")

    @staticmethod
    def _load_dict(path):
        if op.exists(path):
            with open(path, 'rb') as f:
                ret = pickle.load(f)
        else:
            ret = dict()
        return ret

