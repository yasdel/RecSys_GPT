import unittest

from evaluation import *

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

class TestFairnessEval(unittest.TestCase):
    def runTest(self):
        train_df = pd.read_csv('../data/ml-latest-small/train.csv')
        test_df = pd.read_csv('../data/ml-latest-small/test.csv')
        rec_df = pd.read_csv('./toy_user_rec.csv')
        fairnessEvaluator = FairnessEval(train_df, test_df, rec_df)
        res = fairnessEvaluator.evaluate_fairness(['NDCG','Recall','Precision'], save_prefix='.')
        print(res)
        print(fairnessEvaluator.eval_df) #TODO: assert it has all added columns
        self.assertTrue(all(c in fairnessEvaluator.eval_df.columns for c in list(rec_df.columns) + [
                        'is_mainstream','is_active',f'top-{Eval.TOP_K} class',
                        'pop affinity hist','pop affinity rec','pop miscalibration (JS div)',
                        'hist items','hist scores'
                        ]))

suite = unittest.TestSuite()
suite.addTest(TestRecall())
suite.addTest(TestPrecision())
suite.addTest(TestNDCG())
suite.addTest(TestFairnessEval())
runner = unittest.TextTestRunner()
runner.run(suite)
