# -*- coding: utf-8 -*-
"""
Original file for these methods is located at
    https://colab.research.google.com/drive/1Tjdk1wW7nVHSsi5L_Zjgg0jgSLendD4m
"""

import numpy as np
import pandas as pd

class DataCharExtractor:

  INF = 1e6

  @staticmethod
  def coeff_var(col:pd.Series):
    std = col.std()
    mean = col.mean()
    cv = -DataCharExtractor.INF if mean == 0 else std / mean
    return cv

  @staticmethod
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

  @staticmethod
  def gini_user_item(df):
    '''
    Compute Gini coefficient as a measure of inequality in the distribution 
    of interactions across items ('gini_item'), and across users ('gini_user')

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

  @staticmethod
  def log_density(noUsers, noItems, noRatings):
    log_density = np.log10(noRatings/(noUsers * noItems))
    return log_density

  @staticmethod
  def log_shape(noUsers, noItems):
    log_shape = np.log10(noUsers / noItems)
    return log_shape

  @staticmethod
  def popularity_segment_flexibleGroup(ratings_df, proportion_list):
    '''
    Inputs
    ----------
      - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
      - proportion_list: a list of proportions i.e., values within [0,1] indicating the proportion of interactions belonging to each item class label (e.g. short-head, distant-tail).

    Outputs
    ----------
      - ratings_df_res: the input dataframe with a new column i.e., a class label
        showing if each item belongs to 'short-head'(0), 'mid-tail'(1), 'distant-tail'(2) class.

      - items_df: a dataframe in the form (item, popClass)

      - itemIds: list of ids of short-head, mid-tail, and distant tail items
    
    Usage (*dataset needed*)
    ----------
    popularity_segment_flexibleGroup(toy_df, [0.8,0.2])

    '''
    if sum(proportion_list) != 1: 
      raise ValueError('Proportions in `proportion_list` parameter should sum to 1!')
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

  @staticmethod
  def popularity_bias(ratings_df, proportion_list, verbose=False):
    '''
    Inputs
    ----------
      - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
      - proportion_list: a list of proportions i.e., values within [0,1] indicating the proportion of interactions belonging to each item class label (e.g. short-head, distant-tail).

    Outputs
    ----------
      - pop_bias: Popularity bias, computed as the ratio between average interactions of popular items (lowest popClass label) and avg interactions of niche items (highest popClass label)

      - ratings_df_res: the input dataframe with a new column i.e., a class label
        showing if each item belongs to 'short-head'(0), 'mid-tail'(1), 'distant-tail'(2) class.

      - items_df: a dataframe in the form (item, popClass)

      - itemIds: list of ids of short-head, mid-tail, and distant tail items
    
    Usage (*dataset needed*)
    ----------
    popularity_bias(toy_df, [0.8,0.2], verbose=True)

    '''

    if sum(proportion_list) != 1: 
      raise ValueError('Proportions in `proportion_list` parameter should sum to 1!')
    ratings_df_res, items_df, itemIds = popularity_segment_flexibleGroup(ratings_df, proportion_list)
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
      Number of Ratings collected by short-head items: {noRatingShort})
      Number of Ratings collected by mid-tail items: {noRatingMid})
      Number of Ratings collected by distant-tail items: {noRatingDistant})
      Number of short-head items: {noShort})
      Number of mid-tail items: {noMid})
      Number of distant-tail items: {noDistant})
      # R/I (short-head items): {noRatingShort/noShort})
      # R/I (mid-tail items): {noRatingMid/noMid})
      # R/I (distant-tail items): {noRatingDistant/noDistant})
      ----------------------------------
      ''')

    return pop_bias, ratings_df_res, items_df, itemIds

  @staticmethod
  def item_popularity(ratings_df, proportion_list, return_flag_col=False):
    '''
    Inputs
    ----------
      - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
      - proportion_list: a list of proportions i.e., values within [0,1] and summing to 1, 
                        where each value 'i' indicates the proportion of cumulative interactions of items with class label 'i' out of total interactions. 
                        For example, '[0.8,0.2]' indicates that 80% of interactions will belong to items with class label '0' (i.e., popular items, aka 'short-head'). 

    Outputs
    ----------
      - df_w_pop_info: input dataset enriched with item popularity info (users are ignored here). Specifically, info is detailed in 2 columns:
          1. 'popularity score': numerical score defined as the ratio between observed interactions for a single item and total interactions in df_ratings. 
          2. (if return_flag_col=False) 'is_popular': boolean flag indicating whether the item has been classified as popular or not.
      - (if return_flag_col=True) flag_col: a Series with a boolean flag for each item, indicating whether the item has been classified as popular or not.

    Usage (*dataset needed*)
    ----------
    df, flag_col = item_popularity(toy_df, proportion_list=[0.8,0.2], return_flag_col=True) # or use FairnessEval.POP_PROPORTIONS as proportion_list

    Technical notes
    ----------
    Similar steps as for user activity (because they are both based on number of interactions)
    '''
    pop_df = ratings_df[['userId', 'itemId', 'rating']].groupby('itemId')  \
                                                    .size()  \
                                                    .reset_index(name='count')  \
                                                    .sort_values(['count'], ascending=False)
    tot_int = pop_df['count'].sum()
    # compute some popularity score for each item: number of interactions as percentage of total
    pop_df['popularity score'] = [x/tot_int for x in pop_df['count']]
    # divide users into active and non-active, according to the proportion of total interactions from 'proportion_list' parameter
    pop_df['CDF'] = pop_df['count'].cumsum()
    # Threshold for popularity class
    thres = np.rint(proportion_list[0] * tot_int)
    # create new column acting as class label, where True is non-popular and False is popular
    pop_df['is_popular'] = pop_df['CDF'] < thres
    # Add item popularity info to input dataframe
    df_w_pop_info = ratings_df.merge(pop_df, on='itemId')
    df_w_pop_info = df_w_pop_info[['itemId','is_popular','popularity score']].drop_duplicates()
    df_w_pop_info.loc[:,'itemId'] = df_w_pop_info['itemId'].map(str)
    df_w_pop_info = df_w_pop_info.set_index('itemId')

    if return_flag_col: return df_w_pop_info.drop(columns=['is_popular']), df_w_pop_info['is_popular']
    return df_w_pop_info

  @staticmethod
  def user_mainstreaminess(ratings_df, mainstr_thres=0, return_flag_col=False):
    '''
    Inputs
    ----------
      - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
      - mainstr_thres: 
          A threshold used to determine which users are considered 'mainstream' i.e. have preferred popular items in observed interactions.
          Specifically, this threshold is used with a 'mainstreaminess_score' (see output below for further details on this score).

    Outputs
    ----------
      - df_w_mainstr_info: input dataset enriched with user mainstreaminess info (items are ignored here). Specifically, info is detailed in 3 columns:
          1. 'historical popularity affinity': list with proportion of user interactions with popular and unpopular items, respectively. 
                                              Note that ratio between these two elements gives 'mainstreaminess score'
          2. 'mainstreaminess score': numerical score based on user history, defined as the ratio between proportions of popular and unpopular items she has interacted with. 
                                      If denominator is 0 (user has not interacted with any non-popular item), this score is set to 1e5
          3. (if return_flag_col=False) 'is_mainstream': boolean flag indicating whether the user has been classified as mainstream or not.
      - (if return_flag_col=True) flag_col: a Series with a boolean flag for each user, indicating whether the user has been classified as mainstream or not.

    Usage (*dataset needed*)
    ----------
    df, flag_col = user_mainstreaminess(toy_df, mainstr_thres=3, return_flag_col=True) # or use FairnessEval.MAINSTR_THRES as mainstr_thres

    '''

    ratings_df_res, items_df, itemIds = popularity_segment_flexibleGroup(ratings_df, [0.8,0.2])
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
    df_w_mainstr_info = ratings_df.join(hist_pop_affinity, on='userId', how='left')
    df_w_mainstr_info = df_w_mainstr_info[['userId','historical popularity affinity','mainstreaminess score','is_mainstream']].drop_duplicates()
    df_w_mainstr_info = df_w_mainstr_info.set_index('userId')

    if return_flag_col: return df_w_mainstr_info.drop(columns=['is_mainstream']), df_w_mainstr_info['is_mainstream']
    return df_w_mainstr_info

  @staticmethod
  def user_activity(df_ratings, proportion_list, return_flag_col=False):
    '''
    Inputs
    ----------
      - ratings_df: a UIR dataframe in the form (userId, itemId, rating)
      - proportion_list: a list of proportions i.e., values within [0,1] and summing to 1, 
                        where each value 'i' indicates the proportion of cumulative interactions of users with i-th class label out of total interactions. 
                        For example, '[0.8,0.2]' indicates that 80% of interactions will belong to users with class label '0' (i.e., active users). 

    Outputs
    ----------
      - df_w_activ_info: input dataset enriched with user activity info (items are ignored here). Specifically, info is detailed in 2 columns:
          1. 'activity score': numerical score defined as the ratio between observed interactions for a single user and total interactions in df_ratings. 
          2. (if return_flag_col=False) 'is_active': boolean flag indicating whether the user has been classified as active or not.
      - (if return_flag_col=True) flag_col: a Series with a boolean flag for each user, indicating whether the user has been classified as active or not.

    Usage (*dataset needed*)
    ----------
    df, flag_col = user_activity(toy_df, proportion_list=[0.8,0.2], return_flag_col=True) # or use FairnessEval.ACTIV_PROPORTIONS as proportion_list

    '''

    activity_df = df_ratings[['userId', 'itemId', 'rating']].groupby('userId')  \
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
    df_w_activ_info = df_ratings.merge(activity_df, on='userId')
    df_w_activ_info = df_w_activ_info[['userId','is_active','activity score']].drop_duplicates()
    df_w_activ_info = df_w_activ_info.set_index('userId')

    if return_flag_col: return df_w_activ_info.drop(columns=['is_active']), df_w_activ_info['is_active']
    return df_w_activ_info

  @staticmethod
  def extract_fairnessDC(df):
    stats = group_fairness(df)
    kw_mapper = {
        'mainstr': {'long_kw': 'user_mainstr', 'grp': 'ug'},
        'activ':   {'long_kw': 'user_activ',   'grp': 'ug'},
        'pop':     {'long_kw': 'item_pop',     'grp': 'ig'}
    }
    fairDC_dct = dict()
    for k, v in kw_mapper.items():
      stats_df = stats[k]
      long_kw, grp = v['long_kw'], v['grp']
      # Transform boolean df index (False, True) in strings
      stats_df.index = stats_df.index.map(str)
      fairDC_dct.update({
          f'std_{k}_{grp}1': get_from_stats(stats_df, row='False', col_suffix='std'),
          f'avg_{k}_{grp}1': get_from_stats(stats_df, row='False', col_suffix='mean'),
          f'cv_{k}_{grp}1':  get_from_stats(stats_df, row='False', col_suffix='coeff_var'),
          f'std_{k}_{grp}2': get_from_stats(stats_df, row='True',  col_suffix='std'),
          f'avg_{k}_{grp}2': get_from_stats(stats_df, row='True',  col_suffix='mean'),
          f'cv_{k}_{grp}2':  get_from_stats(stats_df, row='True',  col_suffix='coeff_var'),
          f'std_overall_{long_kw}': get_from_stats(stats_df, row='Overall', col_suffix='std'),
          f'avg_overall_{long_kw}': get_from_stats(stats_df, row='Overall', col_suffix='mean'),
          f'cv_overall_{long_kw}':  get_from_stats(stats_df, row='Overall', col_suffix='coeff_var')
      })

    # Derived DCs
    fairDC_dct.update({
        'cv_mainstr disparity (abs)': abs(fairDC_dct['cv_mainstr_ug1'] - fairDC_dct['cv_mainstr_ug2']),
        'cv_activ disparity (abs)': abs(fairDC_dct['cv_activ_ug1'] - fairDC_dct['cv_activ_ug2']),
        'cv_pop disparity (abs)': abs(fairDC_dct['cv_pop_ig1'] - fairDC_dct['cv_pop_ig2']),
    })

    return fairDC_dct

  @staticmethod
  def group_fairness(self):
    '''
    This function computes both group-level and overall statistics (standard deviation, mean, coefficient of variation
    i.e. ratio of standard deviation to the mean) for fairness-oriented Data Characteristics
    i.e., user mainstreamness, user activity, and item popularity.

    Example output
    ------
    #TODO
    '''

    stats = dict()
    stats['mainstr'] = group_stats(dc_char=user_mainstreaminess(self.train_data, MAINSTR_THRES),
                                    flag_col='is_mainstream', score_col='mainstreaminess score')
    stats['activ'] = group_stats(dc_char=user_activity(self.train_data, ACTIV_PROPORTIONS),
                                  flag_col='is_active', score_col='activity score')
    stats['pop'] = group_stats(dc_char=item_popularity(self.train_data, POP_PROPORTIONS),
                                flag_col='is_popular', score_col='popularity score')

    return stats

  @staticmethod
  def group_stats(dc_char, flag_col, score_col):
    df_stats = dc_char.groupby([flag_col], as_index=False) \
                      .agg({score_col:['mean', 'std', coeff_var]})

    df_stats.set_index(flag_col, inplace=True)
    df_stats.columns = [' '.join(col).strip() for col in df_stats.columns.values]
    stats_cols = df_stats.columns

    df_stats.loc['Overall'] = [dc_char[score_col].mean(), dc_char[score_col].std(), coeff_var(dc_char[score_col])]

    return df_stats

  @staticmethod
  def get_from_stats(stats, row, col_suffix):
    x = stats.loc[row, filter(lambda x: x.endswith(col_suffix), stats.columns)]
    try:
      return float(x)
    except Exception: 
      raise ValueError(f'Unexpected results from stats.loc: {x}')

  @staticmethod
  def calculate_user_item_statistics(UIMat_df, base_data_char=True, gini=True, fairness_data_char=False, verbose=True):
      """
      Calculates various statistics from a user-item interaction matrix DataFrame.

      Parameters:
      UIMat_df (DataFrame): A pandas DataFrame with columns 'userId', 'itemId', and 'rating'.
      verbose: if True, prints the computed statistics

      Returns:
      dict: A dictionary containing:
            - number of users
            - number of items
            - average number of ratings per user
            - average number of ratings per item
            - sparsity of the dataset (and log10)
            - shape of the dataset (and log10)
            - Gini coefficients for rating distributions over users and items
      """

      # Calculate the number of users and items
      num_users = UIMat_df['userId'].nunique()
      num_items = UIMat_df['itemId'].nunique()

      num_ratings = len(UIMat_df)

      # Calculate the average number of ratings per user
      ratings_per_user = UIMat_df.groupby('userId')['rating'].count().mean()

      # Calculate the average number of ratings per item
      ratings_per_item = UIMat_df.groupby('itemId')['rating'].count().mean()

      # Create a dictionary of the calculated statistics
      stats = {
          'num_ratings': num_ratings,
          'num_users': num_users,
          'num_items': num_items,
          'avg_ratings_per_user': ratings_per_user,
          'avg_ratings_per_item': ratings_per_item,
      }

      if base_data_char:
        # Calculate the sparsity of the dataset
        sparsity = 100 * (1 - len(UIMat_df) / (num_users * num_items))
        log10_density = DataCharExtractor.log_density(num_users, num_items, num_ratings)
        shape = num_users / num_items
        log10_shape = DataCharExtractor.log_shape(num_users, num_items)
        stats.update({
          'sparsity': sparsity,
          'log_density': log10_density,
          'shape': shape,
          'log_shape': log10_shape
        })

      if gini:
        # Assume 'gini_user_item' is a predefined function that returns Gini coefficients for users and items
        gini_user, gini_item = gini_user_item(UIMat_df)
        stats.update({
          'gini_user': gini_user,
          'gini_item': gini_item
        })

      # if fairness_data_char:

      if verbose:
        print('....................................')
        print(f"Number of ratings: {stats['num_ratings']}")
        print(f"Number of users: {stats['num_users']}")
        print(f"Number of items: {stats['num_items']}")
        print(f"Average number of ratings per user: {stats['avg_ratings_per_user']}")
        print(f"Average number of ratings per item: {stats['avg_ratings_per_item']}")
        print(f"Sparsity: {stats.get('sparsity', '-'):.4f} %")
        print(f"Log10 Density: {stats.get('log_density', '-'):.4f} %")
        print(f"Shape: {stats.get('shape', '-'):.4f} %")
        print(f"Log10 Shape: {stats.get('log_shape', '-'):.4f} %")
        print(f"Gini user: {stats.get('gini_user', '-')}")
        print(f"Gini item: {stats.get('gini_item', '-')}")
        print('....................................')

      return stats
