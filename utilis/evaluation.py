# -*- coding: utf-8 -*-
"""Evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1We_vaa_ebup6xQYP3Mj3URkezgW6uGGk
"""

import numpy as np
import unittest

class Eval:

  def __init__(self, ranked_list, pos_items):
    self.ranked_list = ranked_list
    self.pos_items = pos_items

  @staticmethod
  def recall(ranked_list, pos_items):

    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / pos_items.shape[0] if len(pos_items) else 0.0

    assert 0 <= recall_score <= 1, recall_score
    return recall_score

  @staticmethod
  def precision(ranked_list, pos_items):
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float64) / len(is_relevant) if len(is_relevant) else 0.0

    assert 0 <= precision_score <= 1, precision_score
    return precision_score

  @staticmethod
  def ndcg(ranked_list, pos_items, relevance=None, at=None):

    if relevance is None:
        relevance = np.ones_like(pos_items, dtype=int)
    assert len(relevance) == pos_items.shape[0]

    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in ranked_list[:at]], dtype=np.float32)
    ideal_dcg = Eval.dcg(np.sort(relevance)[::-1])
    rank_dcg = Eval.dcg(rank_scores)
    ndcg_ = rank_dcg / ideal_dcg if rank_dcg != 0.0 else 0.0

    return ndcg_

  @staticmethod
  def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)), dtype=np.float32)


class TestRecall(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(Eval.recall(ranked_list_1, pos_items), 3. / 4))
        self.assertTrue(np.allclose(Eval.recall(ranked_list_2, pos_items), 1.0))
        self.assertTrue(np.allclose(Eval.recall(ranked_list_3, pos_items), 0.0))

class TestPrecision(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])
        self.assertTrue(np.allclose(Eval.precision(ranked_list_1, pos_items), 0.6))
        self.assertTrue(np.allclose(Eval.precision(ranked_list_2, pos_items), 0.8))
        self.assertTrue(np.allclose(Eval.precision(ranked_list_3, pos_items), 0.0))

class TestNDCG(unittest.TestCase):
    def runTest(self):
        pos_items = np.asarray([2, 4, 5, 10])
        pos_relevances = np.asarray([5, 4, 3, 2])
        ranked_list_1 = np.asarray([1, 2, 3, 4, 5])  # rel = 0, 5, 0, 4, 3
        ranked_list_2 = np.asarray([10, 5, 2, 4, 3])  # rel = 2, 3, 5, 4, 0
        ranked_list_3 = np.asarray([1, 3, 6, 7, 8])  # rel = 0, 0, 0, 0, 0
        idcg = ((2 ** 5 - 1) / np.log(2) +
                (2 ** 4 - 1) / np.log(3) +
                (2 ** 3 - 1) / np.log(4) +
                (2 ** 2 - 1) / np.log(5))
        self.assertTrue(np.allclose(Eval.dcg(np.sort(pos_relevances)[::-1]), idcg))
        self.assertTrue(np.allclose(Eval.ndcg(ranked_list_1, pos_items, pos_relevances),
                                    ((2 ** 5 - 1) / np.log(3) +
                                     (2 ** 4 - 1) / np.log(5) +
                                     (2 ** 3 - 1) / np.log(6)) / idcg))
        self.assertTrue(np.allclose(Eval.ndcg(ranked_list_2, pos_items, pos_relevances),
                                    ((2 ** 2 - 1) / np.log(2) +
                                     (2 ** 3 - 1) / np.log(3) +
                                     (2 ** 5 - 1) / np.log(4) +
                                     (2 ** 4 - 1) / np.log(5)) / idcg))
        self.assertTrue(np.allclose(Eval.ndcg(ranked_list_3, pos_items, pos_relevances), 0.0))


suite = unittest.TestSuite()
suite.addTest(TestRecall())
suite.addTest(TestPrecision())
suite.addTest(TestNDCG())
runner = unittest.TextTestRunner()
runner.run(suite)