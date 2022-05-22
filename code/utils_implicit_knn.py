import logging
from typing import Dict, Any, Union, Tuple, List
from datetime import datetime

import implicit
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils_models import (
    get_user_item_matrix,
    build_lookup_array
)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


PREDICT_BATCH_SIZE = 10000
model_registry = {
    'tfidf': implicit.nearest_neighbours.TFIDFRecommender,
    'cos': implicit.nearest_neighbours.CosineRecommender,
    'bm25': implicit.nearest_neighbours.BM25Recommender,
}


def batch_array_sort(array: csr_matrix, 
                     num_candidates:int,
                     predict_batch_size:int):
    """
    Method for batch argsort of large numpy array
    return sorted array - indexes of sorted scores
    """
    # calculate num batches
    num_bathes = int(array.shape[0]/predict_batch_size) + 1
    sorted_arr = np.empty(shape=(0, num_candidates), dtype='int32')

    for batch_ind in tqdm(range(num_bathes)):
        low_ind = batch_ind*predict_batch_size
        high_ind = (batch_ind + 1)*predict_batch_size
        temp_arr = array[low_ind: high_ind].toarray().astype('float32')

        temp_sorted = np.argsort(
                        -temp_arr,
                        axis=1,
                        kind='stable')[:, :num_candidates].astype('int32')
        sorted_arr = np.vstack((sorted_arr, temp_sorted))
        
    return sorted_arr


def implicit_fit_predict(
    data: pd.DataFrame,
    train_start_date_: datetime, 
    train_end_date_: datetime,
    user_item_value: str,
    model_type: str,
    model_params: Dict[str, Any],
    similarity_type: str,
    num_candidates: int,
    user_cnt_unq_items: int,
    item_cnt_unq_users: int,
    return_df: bool = True
) -> Union[pd.DataFrame, List[Tuple[Any, Any]]]:
    """
    data: pd.DataFrame, transactions
    user_item_value: str, 'cnt_articles', 'item_weight', 'price_weight' - use to get item value
    model_type: str, 'tfidf', 'cos', 'bm25'
    similarity_type: str, 'i2i', 'u2u'
    user_cnt_unq_items: int, filtering threshold - min unique cnt of items per user
    item_cnt_unq_users: int, filtering threshold - min unique cnt of users per item
    return_df: bool, return list or dataframe
    """
    # select train data
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    # time decay in days and item_weight
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    temp['price_weight'] = temp['time_coef'] * temp['price']
    
    logging.info('Trans data collected')
    
    # collect user item logs
    user_item_log = temp.groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({user_item_value: 'sum'})\
        .rename(columns={user_item_value: 'value'})
    logging.info('user_item_log.shape = %s', user_item_log.shape)
    
    # filter rare items and users with few items
    temp_item = user_item_log.groupby(['article_id'], as_index=False)\
        .agg({'customer_id': 'count'})\
        .rename(columns={'customer_id': 'cnt_unq_customers'})
    items_to_drop = temp_item[temp_item['cnt_unq_customers'] <= item_cnt_unq_users]['article_id'].tolist()
    logging.info('len(items_to_drop) = %s', len(items_to_drop))

    temp_user = user_item_log.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': 'count'})\
        .rename(columns={'article_id': 'cnt_unq_articles'})
    users_to_drop = temp_user[temp_user['cnt_unq_articles'] <= user_cnt_unq_items]['customer_id'].tolist()
    logging.info('len(users_to_drop) = %s', len(users_to_drop))
    
    user_item_log_filtered = user_item_log[(~user_item_log['customer_id'].isin(users_to_drop)) &
                                           (~user_item_log['article_id'].isin(items_to_drop))].copy()
    logging.info('user_item_log_filtered.shape = %s', user_item_log_filtered.shape)
    
    user_item_log_filtered = [tuple(x) for x in user_item_log_filtered.to_numpy()]
    logging.info('len(user_item_log_filtered) = %s', len(user_item_log_filtered))
    
    # create sparse matrix, user item mappings, lookup array
    user_index, item_index, user_items = get_user_item_matrix(user_item_log_filtered, 
                                                              data_format='cnt')
    index_item = {item_index[item]: item for item in item_index}
    index_user = {user_index[user]: user for user in user_index}
    lookup_arr = build_lookup_array(index_item)
    user_items = user_items.astype('float32')
    
    logging.info('len(user_index) = %s', len(user_index))
    logging.info('len(item_index) = %s', len(item_index))
    logging.info('len(index_item) = %s', len(index_item))
    logging.info('len(index_user) = %s', len(index_user))
    logging.info('user_items.shape = %s', user_items.shape)
    
    # create model
    model = model_registry[model_type](**model_params)
    np.random.seed(7)

    scores = None

    if similarity_type == 'i2i':
        # fit model
        model.fit(user_items)
        # predict items for every user in user_items
        scores = user_items.dot(model.similarity).astype('float32')
    
    if similarity_type == 'u2u':
        # fit model
        model.fit(user_items.T)
        # predict items for every user in user_items
        scores = user_items.T.dot(model.similarity).T.astype('float32')
    
    preds_sorted = batch_array_sort(scores, num_candidates, PREDICT_BATCH_SIZE)
    # transform index to article_id
    preds_sorted = lookup_arr[preds_sorted]
    if return_df:
        # create pandas dataframe
        preds_df = pd.DataFrame([(index_user[ind], preds_sorted[ind])
                                 for ind in range(len(index_user))],
                                columns=['customer_id', 'pred'])
        return preds_df
    else:
        return [(index_user[ind], preds_sorted[ind])
                for ind in range(len(index_user))]
