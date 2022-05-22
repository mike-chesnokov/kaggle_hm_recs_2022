import logging
from typing import Tuple, Dict, List

import pandas as pd
from datetime import datetime, timedelta


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def concat_items(list1: List[int], 
                 list2: List[int],
                 num_candidates: int):
    """Concat 2 lists without repetitions and save the order"""
    list1_set = set(list1)
    items_to_fill = [item for item in list2 if item not in list1_set]
    result = list1 + items_to_fill
    return result[:num_candidates]


def get_dates(test_start_date_: datetime,
              num_days_test: int,
              num_days_train: int) -> Tuple[datetime, datetime, datetime, datetime]:
    """Dates for train and test folds (no valid fold)"""
    test_end_date = test_start_date_ + timedelta(days=num_days_test - 1)
    train_end_date = test_start_date_ - timedelta(days=1)
    train_start_date = train_end_date - timedelta(days=num_days_train)
    return test_start_date_, test_end_date, train_end_date, train_start_date


def get_sub_df_popular(sub_path: str,
                       target: Dict[int, List[int]],
                       preds: List[int]):
    """
    Return stub for submisison popular. 
    Customers and articles are NOT in initial view (with  '0')
    """
    sub_df = pd.read_feather(sub_path)
    sub_df['target'] = sub_df['customer_id'].apply(lambda x: target.get(x, []))
    sub_df['pred'] = [preds for _ in range(sub_df.shape[0])]
    return sub_df


def get_sub_df(sub_path: str,
               target: Dict[int, List[int]],
               preds: pd.DataFrame,
               stub_articles: List[int],
               num_candidates: int):
    """
    Return stub for submisison popular. 
    Customers and articles are NOT in initial view
    
    Params:
        preds pd.DataFrame, "customer_id, pred"
        stub_articles: list, items to fill empty 
    """
    sub_df = pd.read_feather(sub_path)
    sub_df['target'] = sub_df['customer_id'].apply(lambda x: target.get(x, []))
    sub_df = sub_df.merge(preds, how='left', on='customer_id')
    sub_df['pred'] = sub_df['pred'].fillna('').apply(list)
    sub_df['pred'] = sub_df['pred'].apply(lambda x: concat_items(x, stub_articles, num_candidates))
    return sub_df


def get_sub_df_personal_cold(
    sub_path: str,
    target: Dict[int, List[int]],
    preds: pd.DataFrame,
    preds_personal_cold_: pd.DataFrame,
    num_candidates: int
):
    sub_df = pd.read_feather(sub_path)
    sub_df['target'] = sub_df['customer_id'].apply(lambda x: target.get(x, []))
    
    # join personal preds
    sub_df = sub_df.merge(preds, how='left', on='customer_id')
    sub_df['pred'] = sub_df['pred'].fillna('').apply(list)
    
    # join personal cold start
    sub_df = sub_df.merge(preds_personal_cold_, how='left', on=['customer_id'])
    
    sub_df['pred'] = sub_df[['pred', 'personal_cold_start']]\
        .apply(lambda x: concat_items(x[0], x[1], num_candidates), axis=1)
    return sub_df


def get_test_target(data: pd.DataFrame, 
                    test_start_date_: datetime, 
                    test_end_date_: datetime) -> Dict[int, List[int]]:
    """Collect target items to dict: customer_ind -> list of articles"""
    target = data[(data['t_dat'] >= test_start_date_) & 
                  (data['t_dat'] <= test_end_date_)]\
        .groupby(['customer_id'], as_index=False)\
        .agg({'article_id': set})\
        .to_dict(orient='records')
    # transform to dict 
    target = {row['customer_id']: row['article_id'] for row in target}
    return target
