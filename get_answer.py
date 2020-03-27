# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:01:42 2019

@author: cmy
"""
import codecs as cs
import pickle
import time
from math import exp
from random import random
from typing import Dict, Union, Callable, Iterable
from sklearn.externals import joblib

import click
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from corpus import Corpus
import sys
# sys.path.append("..")
# from bert4.similarity import BertSim
from nn_utils import cmp
from similarity import BertSim
import tensorflow as tf
from candidate import Candidate
from os import path as op


class AnswerCandidate(Candidate):
    def __init__(self, entity2relations_dict='data/entity2relations_dict.pkl',
                 seqPair2similarity_dict='data/seqPair2similarity_dict.pkl'):
        self._entity2relations = self._load_dict(entity2relations_dict)
        self._seqPair2similarity = self._load_dict(seqPair2similarity_dict)
        self._similarity_dict_path = seqPair2similarity_dict
        self._relation_paths_dict_path = entity2relations_dict
        self._model = BertSim()
        self._model.mode = tf.estimator.ModeKeys.PREDICT

    def _similarity_of(self, faked, seq):
        k = faked + seq
        if k not in self._seqPair2similarity:
            self._seqPair2similarity[k] = self._model.predict(faked, seq)
        return self._seqPair2similarity[k]

    def _relation_paths_of(self, entity):
        if entity not in self._entity2relations:
            return []
        return self._entity2relations[entity]

    def _candidates_of(self, entity2feats, question):
        answer2feats = {}
        for entity, feats in entity2feats.items():
            relation_paths = self._relation_paths_of(entity)
            if not relation_paths:
                continue
            mention = feats[0]
            for relations in relation_paths:
                answer = (entity, *relations)
                predicates = [spo[1:-1] for spo in relations]
                hypothesis = '的'.join([mention] + predicates)
                feats = [entity, mention, self._similarity_of(hypothesis, question)]
                answer2feats[answer] = feats
        return answer2feats

    def candidates_of(self, subject2feats: Dict[str, list], question: str):
        return self._candidates_of(subject2feats, question)

    def add_candidates_to_corpus(self, corpus: Corpus):
        num_answers = .0
        num_2hop = .0
        num_cover = {
            'all': .0,
            '2hop': .0
        }
        for i, sample in enumerate(corpus):
            question = sample['question']
            gold_answer = sample['gold_tuple']
            gold_entities = sample['gold_entitys']
            subject_linked = sample['subject_linked']
            candidate_answers = self._candidates_of(subject_linked, question)
            num_answers += len(candidate_answers)
            sample['candidate_answer'] = candidate_answers
            ever_cover = False
            for answer in candidate_answers:
                if set(answer).issuperset(gold_answer):
                    ever_cover = True
                    print('* Question: ({}){}\n*\tAnswer: {}'.format(i, question, answer))
                    break
            if ever_cover:
                num_cover['all'] += 1
                if len(gold_answer) <= 3 and len(gold_entities) == 1:
                    num_cover['2hop'] += 1
            if len(gold_answer) <= 3 and len(gold_entities) == 1:
                num_2hop += 1
            # if i >  500 and i % 500 == 0:
            #     print(">>> Caching query dict... <<< ")
            #     self.cache_similarity_query()
            #     self.cache_relation_paths()
        print("* For {}".format(corpus.name))
        print('* Cover ratio in all questions: {:.2f}'.format(num_cover['all'] / len(corpus)))
        print('* Cover ratio in single-entity questions: {:.2f}'.format(num_cover['2hop'] / num_2hop))
        print('* Averaged candidates per question: {:.2f}'.format(num_answers / len(corpus)))
        return corpus

    def cache_similarity_query(self):
        with open(self._similarity_dict_path, 'wb') as f:
            pickle.dump(self._seqPair2similarity, f)

    def cache_relation_paths(self):
        with open(self._relation_paths_dict_path, 'wb') as f:
            pickle.dump(self._entity2relations, f)


class AnswerFilter:
    TYPE_COMPUTE_BEST = Callable[[float, int], float]

    def __init__(self, _dtype='float32', cache_dir='data'):
        self._classifier = linear_model.LogisticRegression(C=1, random_state=1997, class_weight='auto')
        self._transformer = StandardScaler()
        self._dtype = _dtype
        self._cache_dir = cache_dir
        self._valid_predicates = None

    def _preprocess(self, corpus: Corpus, log_stats=True):
        xs = []
        ys = []
        answers = []
        question2predicates = []
        num_right = {
            'all': .0,
            '2hop': .0
        }
        num_2hop = .0
        is_train = corpus.name == 'train'
        for i, sample in enumerate(corpus):
            candidates = sample['candidate_answer']
            gold_predicates = sample['gold_tuple']
            gold_entities = sample['gold_entitys']
            answers.append(sample['answer'])
            has_right = False
            question2predicates.append({
                'gold': gold_predicates,
                'candidate': []
            })
            for answer_predicates, feats in candidates.items():
                if cmp(answer_predicates, gold_predicates) == 0:
                    xs.append(feats[2:])
                    ys.append([1])
                    question2predicates[-1]['candidate'].append(answer_predicates)
                else:
                    prop = random()
                    if prop < 0.5 or not is_train:
                        xs.append(feats[2:])
                        ys.append([0])
                        question2predicates[-1]['candidate'].append(answer_predicates)
                if cmp(answer_predicates, gold_predicates) == 0:
                    has_right = True

            if has_right:
                num_right['all'] += 1
                if len(gold_predicates) <= 3 and len(gold_entities) == 1:
                    num_right['2hop'] += 1
            if len(gold_predicates) <= 3 and len(gold_entities) == 1:
                num_2hop += 1
        xs = np.array(xs, dtype=self._dtype)
        ys = np.array(ys, dtype=self._dtype)
        if log_stats:
            print('* For {}'.format(corpus.name))
            print('* Recall ratio of single-subject questions: {:.2f}'.format(num_right['2hop'] / num_2hop))
            print('\tand the one of all questions: {:.2f}'.format(num_right['all'] / len(corpus)))
        return xs, ys, question2predicates, answers

    def transform(self, x, is_train=False):
        return x
        if is_train:
            self._transformer.fit(x)
            joblib.dump(self._transformer, op.join(self._cache_dir, 'transformer'))
        return self._transformer.transform(x)

    def _cache_model(self):
        with open(op.join(self._cache_dir, 'model/answer_filter.pkl'), 'wb') as f:
            pickle.dump(self._classifier, f)

    @staticmethod
    def retrieve_predicates(prob_labels, question2predicates, top_n):
        predicates_with_prob = []
        for i, predicates in enumerate(question2predicates):
            predicates = predicates['candidate']
            cur_size = len(predicates)
            predicates_with_prob.append(
                sorted(zip(predicates, prob_labels[:cur_size]), key=lambda t: t[1][1], reverse=True)[:top_n])
            prob_labels = prob_labels[cur_size:]
        return predicates_with_prob

    def best_predicates_of(self, corpus: Corpus, top_n):
        if not self._valid_predicates:
            return []
        if corpus.name == 'valid':
            return self._valid_predicates
        xs, ys, question2predicates, *_ = self._preprocess(corpus, False)

        return self.retrieve_predicates(self._classifier.predict_proba(xs).tolist(), question2predicates, top_n)

    @staticmethod
    def recall_of(predicts_list, gold_list):
        num_correct = .0
        num_single = .0
        for predicts, gold in zip(predicts_list, gold_list):
            if len(gold) <= 3:
                num_single += 1
            for predict, prob in predicts:
                if cmp(predict, gold) == 0:
                    num_correct += 1
                    break
        return num_correct / num_single

    def train_n_best(self, name2corpus: Dict[str, Corpus], top_ns, compute_best: TYPE_COMPUTE_BEST):
        train_x, train_y, *_ = self._preprocess(name2corpus['train'])
        *valid_set, question2predicates, answers = self._preprocess(name2corpus['valid'])
        train_x = self.transform(train_x, True)
        valid_x = self.transform(valid_set[0])
        self._classifier.fit(train_x, train_y)
        self._cache_model()
        predict_y = self._classifier.predict_proba(valid_x).tolist()

        best_n = None
        best_val = .0
        print('* For valid')
        for top_n in top_ns:
            predicates_with_prob = self.retrieve_predicates(predict_y, question2predicates, top_n)
            recall = self.recall_of(predicates_with_prob, [it['gold'] for it in question2predicates])
            print('* [Top {}] Recall of single-subject questions: {:.2f}'.format(top_n, recall))
            val = compute_best(recall, top_n)
            if val > best_val:
                best_n = top_n
                best_val = val
                self._valid_predicates = predicates_with_prob
        return best_n


TYPE_CORPUS_DICT = Dict[str, Union[str, Corpus]]


def get_answer_candidates(corpus_dict: TYPE_CORPUS_DICT):
    ac = AnswerCandidate()
    for name, path in corpus_dict.items():
        corpus = ac.add_candidates_to_corpus(Corpus(path, name))
        corpus.save('data/candidate_answer_{}.pkl'.format(name))
        corpus_dict[name] = corpus
    ac.cache_similarity_query()
    # ac.cache_relation_paths()
    return corpus_dict


def get_answer_filtered(corpus_dict: TYPE_CORPUS_DICT):
    assert set(corpus_dict.keys()).issuperset(['train', 'valid'])
    if type(corpus_dict['train']) != Corpus:
        for name, path in corpus_dict.items():
            corpus_dict[name] = Corpus(path, name)

    def compute_best(recall, top_n):
        return recall - top_n / 100

    answer_filter = AnswerFilter()
    top_n = answer_filter.train_n_best(corpus_dict,
                                       [1, 5, 10],
                                       compute_best)
    print('* Best top n: {}'.format(top_n))
    for name, corpus in corpus_dict.items():
        predicates_with_prob = answer_filter.best_predicates_of(corpus, top_n)
        for i, sample in enumerate(corpus):
            filtered_answer = {}
            for predicates, _ in predicates_with_prob[i]:
                feats = sample['candidate_answer'][predicates]
                filtered_answer[predicates] = feats
            sample['filtered_answer'] = filtered_answer
        corpus.save(op.join(op.dirname(corpus.path), 'filtered_answer.{}.pkl'.format(name)))
    return corpus_dict


@click.command()
@click.option('--task', default='all',
              help='available tasks: \n\t[candidates, filtered]')
def main(task):
    corpus2paths = {name: 'data/subject_linked_{}.pkl'.format(name) for name in ['train', 'valid']}
    from functools import reduce
    execute = {
        'all': lambda: reduce(lambda ret, fn: fn(ret), [get_answer_candidates, get_answer_filtered], corpus2paths),
        'candidates': lambda: get_answer_candidates(corpus2paths),
        'filtered': lambda: get_answer_filtered(
            {name: op.join(op.dirname(path), 'candidate_answer_{}.pkl'.format(name)) for name, path in
             corpus2paths.items()}),
    }
    if task in execute:
        execute[task]()
    else:
        print('<<< Unknown Task')


def create_predicates_extractor():
    ac = AnswerCandidate()

    def predicates_extract(subject2feats, question):
        return ac.candidates_of(subject2feats, question)

    return predicates_extract


if __name__ == '__main__':
    main()
