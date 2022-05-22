import logging
from typing import Dict, List, Set

import pandas as pd
from datetime import datetime, timedelta


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_dates_2lvl(
        target_valid_start_date_: datetime,
        num_days_target: int,
        num_days_feat: int):
    """
    Dates for 2nd lvl model for train and valid folds.
    num_days_target: ind, cnt days in target trans
    num_days_feat: ind, cnt days in features trans
    """
    target_valid_end_date = target_valid_start_date_ + timedelta(days=num_days_target - 1)
    
    feat_valid_end_date = target_valid_start_date_ - timedelta(days=1)
    feat_valid_start_date = feat_valid_end_date - timedelta(days=num_days_feat)
    
    target_train_end_date = target_valid_start_date_ - timedelta(days=1)
    target_train_start_date = target_train_end_date - timedelta(days=num_days_target - 1)
    
    feat_train_end_date = target_train_start_date - timedelta(days=1)
    feat_train_start_date = feat_train_end_date - timedelta(days=num_days_feat)
    
    return target_valid_start_date_, target_valid_end_date, \
        feat_valid_start_date, feat_valid_end_date, \
        target_train_start_date, target_train_end_date, \
        feat_train_start_date, feat_train_end_date


def get_target(
        data: pd.DataFrame,
        start_date_: datetime,
        end_date_: datetime
) -> Dict[int, List[int]]:
    """Collect target items to dict: customer_ind -> list of articles"""
    target = data[(data['t_dat'] >= start_date_) & 
                  (data['t_dat'] <= end_date_)]\
        .groupby(['customer_id'], as_index=False)\
        .agg({'article_id': set})\
        .to_dict(orient='records')
    # transform to dict 
    target = {row['customer_id']: row['article_id'] for row in target}
    return target


def get_target_df(
            user_target_items: Dict[str, set],
            users: Set[str] = None,
) -> pd.DataFrame:
    """
    :param users: set, current users, if None - no filtering
    :param user_target_items: dict, user_id -> {item_id1, item_id2, ...}
    """
    logging.debug('get_df STARTED')
    # intersection of all target with current users
    if users is None:
        target_data = [
            (user, item)
            for user in user_target_items
            for item in user_target_items[user]
        ]
    else:
        target_data = [
            (user, item)
            for user in user_target_items
            for item in user_target_items[user]
            if user in users
        ]
    # create dataframe
    target_df = pd.DataFrame(target_data, columns=['customer_id', 'article_id'])
    target_df['target'] = 1

    # change dtype to reduce memory
    target_df_dtypes = {
        'customer_id': 'int32',
        'article_id': 'int32',
        'target': 'int8'
    }
    target_df = target_df.astype(dtype=target_df_dtypes)

    logging.debug('target_df shape = %s', target_df.shape)
    logging.debug('CNT customer_id in target = %s', target_df['customer_id'].nunique())
    logging.debug('CNT article_id in target = %s', target_df['article_id'].nunique())
    logging.debug('get_df DONE')
    return target_df
