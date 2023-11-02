# -*- coding: utf-8 -*-
"""DC_Extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tjdk1wW7nVHSsi5L_Zjgg0jgSLendD4m
"""

# from https://github.com/oliviaguest/gini/blob/master/gini.py
def gini(array):
  '''
  Calculate the Gini coefficient of a numpy array, according to formula http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
  from http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
  Taken the input array:
  - cast values to float
  - shift them so that they are positive
  - ensure there is not any 0
  - sort values
  - compute Gini index

  Examples
  ---------
  gini(np.array([.2,.2,.2,.2,.2,0])) = 0.1666
  gini(np.array([1,0,0,0,0,0])) = 0.8333

  '''
  # All values are treated equally, arrays must be 1d:
  array = array.flatten()
  array = array.astype(np.float64)
  if np.amin(array) < 0:
      # Values cannot be negative:
      array -= np.amin(array)
  # Values cannot be 0:
  array += 0.0000001
  # Values must be sorted:
  array = np.sort(array)
  # Index per array element:
  index = np.arange(1,array.shape[0]+1)
  # Number of array elements:
  n = array.shape[0]
  # Gini coefficient:
  return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def gini_user_item(df):
  '''
  Compute Gini coefficient measuring inequality in the distribution of interactions
  across items ('gini_item'), and across users ('gini_user')

  Notes
  --------
  Potential shortcomings for Gini index (in general):
  - it does not tell the shape of inequality across distribution,
    considering geometric interpretation of 2*(area between equality diagonal and Lorenz curve),
    thus very different distributions can result in identical Gini coefficients
  - it does not show variations among subgroups within the distribution
    e.g. the distribution of interactions across user mainstreaminess, gender (if known)

  '''
  interactions_per_user = df.userId.value_counts(normalize=True).to_numpy()
  interactions_per_item = df.itemId.value_counts(normalize=True).to_numpy()
  gini_user = gini(interactions_per_user)
  gini_item = gini(interactions_per_item)

  return gini_user, gini_item

import pandas as pd
import numpy as np
def popularity_segment_flexibleGroup(ratings_df, proportion_list):
  '''
  Inputs
  ----------
    - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
    - proportion_list: a list of proportions i.e., values within [0,1] indicating the proportion of a item class label.

  Outputs
  ----------
    - ratings_df_res: the input dataframe with a new column i.e., a class label
      showing if each item belongs to 'short-head'(0), 'mid-tail'(1), 'distant-tail'(2) class.

    - items_df: a dataframe in the form (item, popClass)

    - itemIds: list of ids of short-head, mid-tail, and distant tail items

  '''

  # produces the count matrix for items
  items_df = ratings_df[['userId', 'itemId', 'rating']].groupby('itemId') \
                                .size() \
                                .reset_index(name='count') \
                                .sort_values(['count'], ascending=False)
  # define the thresholds for popularity classes
  tmp = items_df.copy()
  nInt = tmp['count'].sum()
  shortThr = np.rint(proportion_list[0] * nInt)
  if len(proportion_list) == 3:
    midThr = np.rint(proportion_list[1] * nInt) + shortThr
  else:
    midThr = None
  # divide items into short_head, mid_tail (and distant_tail, if specified in the 'proportion_list' parameter)
  tmp['CDF'] = tmp['count'].cumsum()

  short_head = tmp.loc[tmp['CDF'].lt(shortThr),'itemId']
  if midThr is not None:
    mid_tail = tmp.loc[tmp['CDF'].lt(midThr) & ~tmp['itemId'].isin(short_head),'itemId']
    distant_tail = tmp.loc[~(tmp['itemId'].isin(mid_tail) | tmp['itemId'].isin(short_head)),'itemId']
    conditions = [items_df['itemId'].isin(short_head), items_df['itemId'].isin(mid_tail), items_df['itemId'].isin(distant_tail)]
    choices = [0, 1, 2]
    items_df['popClass'] = np.select(conditions, choices, default=2)
  else:
    distant_tail = tmp.loc[~tmp['itemId'].isin(short_head),'itemId']
    items_df['popClass'] = np.where(items_df['itemId'].isin(short_head), 0, 1)

  ratings_df_res = ratings_df.merge(items_df, on='itemId')

  # build itemIds
  itemIdsShort = items_df[items_df['popClass'] == 0].sort_values('count')['itemId'].tolist()
  itemIdsMid = items_df[items_df['popClass'] == 1].sort_values('count')['itemId'].tolist()
  itemIdsDistant = items_df[items_df['popClass'] == 2].sort_values('count')['itemId'].tolist()
  itemIds = [itemIdsShort, itemIdsMid, itemIdsDistant]

  return ratings_df_res, items_df, itemIds

def popularity_bias(df, pop_proportion, verbose=False):
  ratings_df_res, items_df, itemIds = popularity_segment_flexibleGroup(df, pop_proportion)
  noRatingShort = (ratings_df_res['popClass'] == 0).sum()
  noRatingMid = (ratings_df_res['popClass'] == 1).sum()
  noRatingDistant = (ratings_df_res['popClass'] == 2).sum()
  noShort = (items_df['popClass'] == 0).sum()
  noMid = (items_df['popClass'] == 1).sum()
  noDistant = (items_df['popClass'] == 2).sum()
  r_i_ratio_head = noRatingShort/noShort
  r_i_ratio_tail = noRatingDistant/noDistant if noDistant!=0 else noRatingMid/noMid
  pop_bias = r_i_ratio_head / r_i_ratio_tail

  if verbose:
    print(f'''
    ----------------------------------
    Number of Ratings collected by short-head items: {noRatingShort}')
    Number of Ratings collected by mid-tail items: {noRatingMid}')
    Number of Ratings collected by distant-tail items: {noRatingDistant}')
    Number of short-head items: {noShort}')
    Number of mid-tail items: {noMid}')
    Number of distant-tail items: {noDistant}')
    # R/I (short-head items): {noRatingShort/noShort}')
    # R/I (mid-tail items): {noRatingMid/noMid}')
    # R/I (distant-tail items): {noRatingDistant/noDistant}')
    ----------------------------------
    ''')

  return pop_bias
# 
# """
# # Example
# %%writefile toy_df.csv
# UserId,ItemId,rating
# 1,3,1
# 1,4,1
# 1,5,1
# 1,7,1
# 1,8,1
# 2,9,1
# 2,3,1
# 2,4,1
# 2,5,1
# 2,70,1
# 3,3,1
# 3,4,1
# 3,60,1
# 3,99,1
# 3,40,1
# 4,3,1
# 4,4,1
# 4,5,1
# 4,70,1
# 4,90,1
# 5,3,1
# 5,4,1
# 5,5,1
# 5,37,1
# 5,39,1
# 6,3,1
# 6,4,1
# 6,5,1
# 6,90,1
# 6,30,1
# 
# toy_df = pd.read_csv('toy_df.csv', header=0, names=["userId","itemId","rating"])
# pop_proportion = [0.5, 0.4,0.1]
# popularity_bias(toy_df, pop_proportion, verbose=True)
# -->
#    userId  itemId  rating  count  popClass
# 0       1       3       1      6         0
# 1       2       3       1      6         0
# 2       3       3       1      6         0
# 3       4       3       1      6         0
# 4       5       3       1      6         0
# 
#     ----------------------------------
#     Number of Ratings collected by short-head items: 12')
#     Number of Ratings collected by mid-tail items: 14')
#     Number of Ratings collected by distant-tail items: 4')
#     Number of short-head items: 2')
#     Number of mid-tail items: 8')
#     Number of distant-tail items: 4')
#     # R/I (short-head items): 6.0')
#     # R/I (mid-tail items): 1.75')
#     # R/I (distant-tail items): 1.0')
#     ----------------------------------
# 6.0
# """
# '' # to avoid echo in output

def log_density(noUsers, noItems, noRatings):
  log_density = np.log10(noRatings/(noUsers * noItems))
  return log_density

def log_shape(noUsers, noItems):
  log_shape = np.log10(noUsers / noItems)
  return log_shape

def user_mainstreaminess(df, mainstr_thres=0, return_groups=False):
  '''
  Tester (*dataset needed*)
  --------
  user_activity(toy_df,  mainstr_thres=3, return_groups=True)

  '''
  ratings_df_res, items_df, itemIds = popularity_segment_flexibleGroup(df, [0.8,0.2])
  # group by user and count items with popClass == i (0 is short-head, 2 is distant tail)
  hist_pop_affinity = pd.crosstab(index=ratings_df_res['userId'], columns=ratings_df_res['popClass'], normalize='index')

  lst, mainstr_scores = [], []
  hist_pop_affinity.columns = hist_pop_affinity.columns.astype(str)
  for a, b in zip(hist_pop_affinity['0'], hist_pop_affinity['1']):
    lst.append([a,b])
    mainstr_scores.append(a/b if b != 0 else 100000)
  hist_pop_affinity.loc[:,'historical popularity affinity'] = [str(x) for x in lst]
  hist_pop_affinity.loc[:,'mainstreaminess score'] = mainstr_scores
  hist_pop_affinity.loc[:,'is_mainstream'] = hist_pop_affinity['mainstreaminess score'] >= mainstr_thres
  hist_pop_affinity.drop(columns=['0','1'], inplace=True)
  df = df.join(hist_pop_affinity, on='userId', how='left')
  df = df[['userId','historical popularity affinity','mainstreaminess score','is_mainstream']]
  df = df.drop_duplicates()
  df = df.set_index('userId')

  if return_groups: return df, df['is_mainstream']
  return df

def user_activity(df, proportion_list, return_groups=False):
  '''
  Tester (*dataset needed*)
  --------
  user_activity(toy_df, [0.8, 0.2], return_groups=True)

  '''
  activity_df = df[['userId', 'itemId', 'rating']].groupby('userId')  \
                                                  .size()  \
                                                  .reset_index(name='count')  \
                                                  .sort_values(['count'], ascending=False)
  tot_int = activity_df['count'].sum()
  # Compute some activity score for each user: number of interactions as percentage of total
  activity_df.loc[:,'activity score'] = [x/tot_int for x in activity_df['count']]
  # Divide users into active and non-active, according to the proportion of total interactions from 'proportion_list' parameter
  activity_df.loc[:,'CDF'] = activity_df['count'].cumsum()
  # Threshold for activity class
  thres = np.rint(proportion_list[0] * tot_int)
  activity_df.loc[:,'is_active'] = activity_df['CDF'] < thres
  df = df.merge(activity_df, on='userId')
  df = df[['userId','is_active','activity score']]
  df = df.drop_duplicates()
  df = df.set_index('userId')

  if return_groups: return df, df['is_active']
  return df
