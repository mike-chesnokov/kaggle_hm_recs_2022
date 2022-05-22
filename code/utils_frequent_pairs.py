from typing import Dict, List
from itertools import combinations
from collections import Counter

import pandas as pd
from datetime import datetime
from tqdm import tqdm


def process_pair(result_dict_, pair, value):
    """util for pair processing"""
    if pair[0] not in result_dict_:
        result_dict_[pair[0]] = {}
    if pair[1] not in result_dict_[pair[0]]:
        result_dict_[pair[0]][pair[1]] = 0
    # increment
    result_dict_[pair[0]][pair[1]] += value
    
    return result_dict_


def get_frequent_pairs(data: pd.DataFrame, 
                    train_start_date_: datetime, 
                    train_end_date_: datetime,
                    num_candidates:int) -> Dict[int, List[int]]:
    """
    calculate num_candidates or less most frequent pairs to items
    """
    temp = data[(data['t_dat'] >= train_start_date_) & 
                   (data['t_dat'] <= train_end_date_)].copy()
    # logging.info('article_id nunique = %s', temp['article_id'].nunique())
    
    # collect buckets from transactions
    buckets = temp\
        .groupby(['t_dat', 'customer_id', 'sales_channel_id'], as_index=False)\
        .agg({'article_id': set})\
        .rename(columns={'article_id': 'bucket'})

    buckets['bucket_size'] = buckets['bucket'].apply(lambda x: len(x))
    # logging.info('buckets.shape = %s', buckets.shape)

    # drop buckets with 1 item
    buckets = buckets[buckets['bucket_size'] > 1]
    # logging.info('Drop 1 items buckets, buckets.shape = %s', buckets.shape)

    # create all pairs from buckets items
    buckets['pairs'] = buckets['bucket'].apply(lambda x: [pair for pair in combinations(x, 2)])
    
    # flatten pairs
    pairs_temp = buckets['pairs'].tolist()
    pairs = []
    for obj in pairs_temp:
        pairs.extend(obj)
    # logging.info('len(pairs) = %s', len(pairs))
    
    # calculate frequency
    counter = Counter(pairs)
    temp_result_dict = {}
    for pair in tqdm(counter):
        temp_result_dict = process_pair(temp_result_dict, pair, counter[pair])
        temp_result_dict = process_pair(temp_result_dict, pair[::-1], counter[pair])
    # logging.info('len(temp_result_dict) = %s', len(temp_result_dict))
    
    # collect dict item -> list of paired items
    frequent_pairs = {}
    for item in tqdm(temp_result_dict):
        inner_dict = temp_result_dict[item]
        inner_dict_sorted = sorted(inner_dict.items(), key=lambda x: x[1], reverse=True)[:12]
        frequent_pairs[item] = [row[0] for row in inner_dict_sorted]
    # logging.info('len(frequent_pairs) = %s', len(frequent_pairs)) 
    
    return frequent_pairs


def concat_pred_freq_items(pred: List[int], 
                           frequent_pairs: Dict[int, List[int]],
                           num_candidates: int) -> List[int]:
    """Join pred items and frequent pairs (if pair exists)"""
    pred_len = len(pred)
    cnt_items_to_fill = num_candidates - pred_len
    items_to_fill = []
        
    ind = 0
    
    while len(items_to_fill) < cnt_items_to_fill and ind < num_candidates * 5:
        #  for cases of cnt items in pred < num_candidates/2
        # loop will go through pred list second time
        ind_ = ind % pred_len
        # check if any frequent item exists
        if pred[ind_] in frequent_pairs:
            paired_items = frequent_pairs[pred[ind_]]
            # iterate over frequent pairs
            for candidate in paired_items:
                if candidate not in pred and candidate not in items_to_fill:
                    items_to_fill.append(candidate)
                    break
        ind += 1
    return pred + items_to_fill


def collect_freq_items(pred: List[int], 
                       frequent_pairs: Dict[int, List[int]],
                       num_candidates: int) -> List[int]:
    """Collect frequent pairs for pred items AND NOT JOIN THEM"""
    pred_len = len(pred)
    items_to_collect = []
    
    ind = 0
    while len(items_to_collect) < num_candidates and ind < num_candidates * 5:
        # for cases of cnt items in pred < num_candidates
        # loop will go through pred list second time
        # if we have passed through pred 5 times and didn't get items_to_collect -
        # break the loop
        ind_ = ind % pred_len
        # check if any frequent item exists
        if pred[ind_] in frequent_pairs:
            # get pairs for current item
            paired_items = frequent_pairs[pred[ind_]]
            # iterate over frequent pairs and chose candidates
            for candidate in paired_items:
                if candidate not in pred and candidate not in items_to_collect:
                    items_to_collect.append(candidate)
                    break
        ind += 1
    return items_to_collect
