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

class FairnessEval:
  
  MAINSTR_THRES = 3
  ACTIV_PROPORTIONS = [0.8, 0.2]
  POP_PROPORTIONS = [0.8, 0.2]


  def __init__(self, train_data, user_recommendations):
    self.train_data = train_data
    self.user_rec = user_recommendations


  def aggregate_metrics(eval_df, metrics_cols):
    '''
    Inputs
    ----------
      - eval_df: a dataframe with both recommended items and user-level metrics, in the form <userId>, <itemIds>, <metric1>, <metric2>, [...]
      - metrics_cols: the list of columns of eval_df containing user-level metrics. For each of these columns, aggregations will be computed.

    Outputs
    ----------
      - aggregation: Dict with all the computed aggregation functions of each metric i.e. its mean value, and disparity of mean value between two user/item groups.
                     It can also be returned as a pd.DataFrame with `pd.DataFrame.from_dict(aggregation, orient='index')`,
                     or as a pd.Series with `pd.Series(aggregation)`
    '''
    # Mean value of the metric
    aggregation = {
        f'Mean {mt}': eval_df[mt].mean()
        for mt in metrics_cols
    }
    # C-Fairness: Disparity in accuracy btw user groups (activity & mainstreaminess)
    for mt in metrics_cols:
      aggregation.update({
          f'Disparity in {mt} for user activity':
          eval_df.groupby('is_active')[mt].agg(np.mean).diff().iloc[1:].abs().squeeze(),
          f'Disparity in {mt} for user mainstreaminess':
          eval_df.groupby('is_mainstream')[mt].agg(np.mean).diff().iloc[1:].abs().squeeze()
      })
    # P-Fairness: Disparity in exposure/visibility btw item groups (popularity)
    class_visibility = pd.Series(eval_df[f'top-{Eval.TOP_K} class'].sum()).value_counts(normalize=True)
    print(class_visibility) # just to debug
    aggregation.update({
        f'Disparity in Visibility@{Eval.TOP_K} for item popularity':
        class_visibility.diff().iloc[1:].abs().squeeze(),
        # f'Disparity in Exposure@{Eval.TOP_K} for item popularity': ...
    })
    return aggregation
