"""
Original file for class Eval is located at
    https://colab.research.google.com/drive/1We_vaa_ebup6xQYP3Mj3URkezgW6uGGk
"""

import numpy as np

class Eval:

  TOP_K = 50
  
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


import pandas as pd
import os
import ast
from dc_extraction import DataCharExtractor
# from dc_extraction import user_activity, user_mainstreaminess, item_popularity
import logging
from setup_logger import logger

class FairnessEval:
  
  MAINSTR_THRES = 3
  ACTIV_PROPORTIONS = [0.8, 0.2]
  POP_PROPORTIONS = [0.8, 0.2]


  def __init__(self, train_data, test_data, user_recommendations):
    self.train_data = train_data
    self.test_data  = test_data
    self.user_rec   = user_recommendations
    self.sanitize_input_data()
    
    self.activ_col   = None
    self.mainstr_col = None
    self.pop_col     = None
    self.eval_df     = None
    
    self.test_items_by_user = test_data.groupby('userId').agg(list)
    self.test_items_by_user.columns = ['itemIds'] + list(self.test_items_by_user.columns[1:])


  def add_accuracy_metrics(self):
    '''
    Outputs
    ----------
      - eval_df: a dataframe with both recommended items and user-level metrics, in the form <userId>, <itemIds>, <metric1>, <metric2>, [...]
    '''
    
    assert 'userId' in self.user_rec, self.user_rec.columns
    logger.info('Computing accuracy metrics for each user in self.user_rec')
    # For each user in self.user_rec, compute all accuracy metrics
    user_metrics_df = self.user_rec.apply(self.user_level_accuracy_metrics, axis=1)
    assert type(user_metrics_df)==pd.DataFrame, f'user_metrics_df has type {type(user_metrics_df)}'
    self.eval_df = self.user_rec.merge(user_metrics_df, on='userId')

    return self # to make it chainable


  def add_membership_info(self, save_prefix=None):
    self.get_user_activity_membership(save_prefix=save_prefix)
    self.get_user_mainstreaminess_membership(save_prefix=save_prefix)
    logger.info('Adding user membership info: activity and mainstreaminess')
    self.eval_df = self.eval_df.merge(self.activ_col.to_frame(), on='userId')
    self.eval_df = self.eval_df.merge(self.mainstr_col.to_frame(), on='userId')

    self.get_item_popularity_membership(save_prefix=save_prefix)
    logger.info('Adding item membership info: popularity')
    self.eval_df[f'top-{Eval.TOP_K} class'] = self.eval_df['itemIds'].map(lambda lst: [self.pop_col[int(i)] for i in lst])
    
    return self # to make it chainable


  def add_popularity_miscalibration(self):
    '''
    Computes JS divergence
    '''
    from scipy.spatial import distance
    self.add_user_history()
    logger.info('Computing user-level popularity miscalibration (JS divergence)')
    popAffinity_hist = self.eval_df['hist class'].map(lambda lst: [lst.count(True)/len(lst), lst.count(False)/len(lst)])
    popAffinity_rec = self.eval_df[f'top-{Eval.TOP_K} class'].map(lambda lst: [lst.count(True)/len(lst), lst.count(False)/len(lst)])
    self.eval_df['pop affinity hist'] = popAffinity_hist
    self.eval_df['pop affinity rec'] = popAffinity_rec
    # Similar to UPD metric (but on single user-level, without group aggregation). UPD is at https://dl.acm.org/doi/pdf/10.1145/3450613.3456821
    self.eval_df['pop miscalibration (JS div)'] = [distance.jensenshannon(h,r) for h,r in zip(popAffinity_hist,popAffinity_rec)]

    return self # to make it chainable


  def add_user_history(self):
    logger.info('Adding user history i.e. set of past interactions with items, together with popularity labels of those items')
    user_hist = self.train_data.groupby('userId').agg(list)
    user_hist.columns = ['hist items', 'hist scores']
    pop_col = self.get_item_popularity_membership()
    user_hist['hist class'] = user_hist['hist items'].map(lambda hist: [pop_col[item] for item in hist])
    self.eval_df = self.eval_df.merge(user_hist, on='userId')
    # return self # to make it chainable


  def aggregate_metrics(self, metrics_cols):
    '''
    Inputs
    ----------
      - metrics_cols: list of columns of self.eval_df containing user-level metrics. For each of these columns, aggregations will be computed.

    Outputs
    ----------
      - aggregation: Dict with all the computed aggregation functions of each metric i.e. its mean value, and disparity of mean value between two user/item groups.
                     It can be transformed to pandas Series with `pd.Series(aggregation)`
    '''

    # Mean value of the metric
    aggregation = {
        f'Mean {mt}': self.eval_df[mt].mean()
        for mt in metrics_cols
    }
    # C-Fairness: Disparity in accuracy btw user groups (activity & mainstreaminess)
    for mt in metrics_cols:
      aggregation.update({
          f'Disparity in {mt} for user activity':
          self.eval_df.groupby('is_active')[mt].agg('mean').diff().iloc[1:].abs().squeeze(),
          f'Disparity in {mt} for user mainstreaminess':
          self.eval_df.groupby('is_mainstream')[mt].agg('mean').diff().iloc[1:].abs().squeeze()
      })
    # P-Fairness: Disparity in exposure/visibility btw item groups (popularity)
    class_visibility = pd.Series(self.eval_df[f'top-{Eval.TOP_K} class'].sum()).value_counts(normalize=True)
    aggregation.update({
        f'Disparity in Visibility@{Eval.TOP_K} for item popularity':
        class_visibility.diff().iloc[1:].abs().squeeze(),
        # f'Disparity in Exposure@{Eval.TOP_K} for item popularity': ...
    })
    return aggregation


  def evaluate_fairness(self, metrics_cols=None, save_prefix=None):
    if metrics_cols: 
      metrics_cols = [mt if '@' in mt else f'{mt}@{Eval.TOP_K}' for mt in metrics_cols]
    os.makedirs(save_prefix, exist_ok=True)
    fairness_metrics = self.add_accuracy_metrics()  \
      .add_membership_info(save_prefix=save_prefix) \
      .add_popularity_miscalibration()              \
      .aggregate_metrics(metrics_cols)

    if save_prefix: pd.Series(fairness_metrics).to_csv(os.path.join(save_prefix, 'fairness_metrics.csv'))

    return fairness_metrics


  def get_item_popularity_membership(self, save_prefix=None):
    if self.pop_col is None: # it will be executed only first time this method is called, like a singleton
      logger.info('Computing item popularity labels, mapping each item to one class (either popular or unpopular)')
      _, pop_col = DataCharExtractor.item_popularity(self.train_data, proportion_list=FairnessEval.POP_PROPORTIONS, return_flag_col=True)
      pop_col.index = pop_col.index.astype(int)
      logger.info('Checking consistency of item popularity mappings with train data')
      assert set(pop_col.index) == set(self.train_data.itemId), \
          f"Items from pop_col are not the same as items in train set. Here are unknown items of pop_col, not present in train data\n{set(pop_col.index).difference(set(self.train_data.itemId))}"
      logger.info('Handling unknown items of test data: setting them as unpopular')
      test_unknown_items = set(self.test_data.itemId).difference(set(self.train_data.itemId))
      for i in test_unknown_items: pop_col[i] = False
      logger.info('Handling possible allucinations (non-existing items in user recommendations): setting them as unpopular')
      rec_unknown_items = set(self.eval_df.itemIds.sum()).difference(set(self.train_data.itemId)).difference(set(self.test_data.itemId))
      for i in rec_unknown_items: pop_col[i] = False
      self.pop_col = pop_col
      if save_prefix: pop_col.to_csv(os.path.join(save_prefix, 'item_popularity.csv'))

    return self.pop_col


  def get_user_activity_membership(self, save_prefix=None):
    if self.activ_col is None: # it will be executed only first time this method is called, like a singleton
      logger.info('Computing user activity labels, indicating whether a user is active or not')
      _, activ_col = DataCharExtractor.user_activity(self.train_data, proportion_list=FairnessEval.ACTIV_PROPORTIONS, return_flag_col=True)
      logger.info('Checking consistency of user activity mapping with train data')
      assert set(activ_col.index) == set(self.train_data.userId), \
        f"Users from {activ_col.name} are not the same as users in train set. Here are unknown users of {activ_col.name}, not present in train data\n{set(activ_col.index).difference(set(self.train_data.userId))}"
      activ_col.index = activ_col.index.astype(int)
      self.activ_col = activ_col
      if save_prefix: activ_col.to_csv(os.path.join(save_prefix, 'user_activity.csv'))

    return self.activ_col


  def get_user_mainstreaminess_membership(self, save_prefix=None):
    if self.mainstr_col is None: # it will be executed only first time this method is called, like a singleton
      logger.info('Computing user mainstreaminess labels, indicating whether a user is mainstream-oriented or not')
      _, mainstr_col = DataCharExtractor.user_mainstreaminess(self.train_data, mainstr_thres=FairnessEval.MAINSTR_THRES, return_flag_col=True)
      logger.info('Checking consistency of mainstreaminess mapping with train data')
      assert set(mainstr_col.index) == set(self.train_data.userId), \
        f"Users from {mainstr_col.name} are not the same as users in train set. Here are unknown users of {mainstr_col.name}, not present in train data\n{set(mainstr_col.index).difference(set(self.train_data.userId))}"
      mainstr_col.index = mainstr_col.index.astype(int)
      self.mainstr_col = mainstr_col
      if save_prefix: mainstr_col.to_csv(os.path.join(save_prefix, 'user_mainstreaminess.csv'))

    return self.mainstr_col


  def sanitize_input_data(self):
    if not all(self.test_data.userId.isin(self.train_data.userId)):
      raise ValueError(f'There are unknown users in test_data: {set(self.test_data.userId).difference(set(self.train_data.userId))}')
    if not all(self.user_rec.userId.isin(self.test_data.userId)):
      raise ValueError(f'There are unknown users in user_recommendations: {set(self.user_rec.userId).difference(set(self.test_data.userId))}')
    if not all(self.test_data.itemId.isin(self.train_data.itemId)):
      logger.warning(f'Number of unknown items in test_data: {len(set(self.test_data.itemId).difference(set(self.train_data.itemId)))}')
    if any(type(rec_list)==str for rec_list in self.user_rec.itemIds):
      logger.warning('Converting passed user recommendations from str to list type')
      self.user_rec.loc[:,'itemIds'] = self.user_rec.itemIds.map(ast.literal_eval)
    if any(self.user_rec.itemIds.map(len) == 0):
      logger.warning(f'Removing {(self.user_rec.itemIds.map(len)==0).value_counts()} users with empty recommendation list')
      self.user_rec = self.user_rec.loc[self.user_rec['itemIds'].map(len) > 0]
    if not all(i in pd.concat([self.train_data.itemId,self.test_data.itemId]).values for i in self.user_rec.itemIds.sum()):
      logger.warning(f'Unrecognized items in user_recommendations: {set(self.user_rec.itemIds.sum()).difference(set(self.train_data.itemId)).difference(set(self.test_data.itemId))}')


  def user_level_accuracy_metrics(self, row:pd.Series):
    '''
    Outputs
    ----------
      - pd.Series with accuracy metrics (NDCG, Recall, and Precision) for the user in the row. 
        Note that these metrics will be expanded to columns when used in pd.DataFrame.apply()
    '''
    user_id = row['userId']
    rec_items = row['itemIds']
    test_items = self.test_items_by_user.at[user_id, 'itemIds']
    # Convert test_items to a NumPy array
    test_items = np.array(test_items)
    # If the length of rec_items is greater than k, truncate to first k items
    if len(rec_items) > Eval.TOP_K:
        rec_items = rec_items[:Eval.TOP_K]
    # if len(rec_items) > len(test_items):
    #   logger.warning(f'There are more recommended items ({len(rec_items)}) than positive items ({len(test_items)})')
    user_metrics = {
      'userId': user_id,
      f'NDCG@{Eval.TOP_K}': Eval.ndcg(rec_items, test_items),
      f'Recall@{Eval.TOP_K}': Eval.recall(rec_items, test_items),
      f'Precision@{Eval.TOP_K}': Eval.precision(rec_items, test_items),
    } 
    return pd.Series(user_metrics)
