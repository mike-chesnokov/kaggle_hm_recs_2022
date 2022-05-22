import logging
from typing import Dict, Any, Tuple

import pandas as pd
from datetime import datetime, timedelta

from utils_implicit_knn import implicit_fit_predict
from utils_validation_2lvl import ( 
    get_target,
    get_target_df
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



INTERACTION_DTYPES = {
    'item_weight': 'float32',
    'days_diff': 'float32',
    't_dat': 'int32',
    'cnt_articles': 'float32',
    'price': 'float32',
    'sales_channel_id': 'float32',
}


def build_candidates(
    user_recs_list,
    use_item_rank=True
) -> pd.DataFrame:
    """
    return candidates_df: pd.DataFrame ['customer_id', 'article_id', 'item_rank'] or
        ['customer_id', 'article_id']
    """
    logging.info('build_candidates STARTED')
    # transform to pd.Dataframe
    if use_item_rank:
        candidates = [(user, item, rank)
                      for user, rec_items in user_recs_list
                      for rank, item in enumerate(rec_items, 1)]
        candidates_df = pd.DataFrame(candidates,
                                     columns=['customer_id', 'article_id', 'item_rank'])
        logging.info('candidates_df shape = %s', candidates_df.shape)

        # change dtype to reduce memory
        candidates_df = candidates_df.astype(dtype={'customer_id': 'int32',
                                                    'article_id': 'int32',
                                                    'item_rank': 'int32'})
    else:
        candidates = [(user, item)
                      for user, rec_items in user_recs_list
                      for item in rec_items]
        candidates_df = pd.DataFrame(candidates,
                                     columns=['customer_id', 'article_id'])
        logging.info('candidates_df shape = %s', candidates_df.shape)

        # change dtype to reduce memory
        candidates_df = candidates_df.astype(dtype={'customer_id': 'int32',
                                                    'article_id': 'int32'})
    logging.info('build_candidates DONE')
    return candidates_df


def features_process(
    features_: pd.DataFrame,
    feature_prefix: str
) -> pd.DataFrame:
    """
    Method for simple features_ process:
        - flattern pandas column names
        - transform dtypes
        - add feature prefix to names
    """
    features = features_.copy()
    
    # flattern the pandas column names
    column_names = zip(features.columns.get_level_values(0), 
                       features.columns.get_level_values(1))
    features.columns = [lvl1 + '_' + lvl2 if lvl2 != '' else lvl1 for lvl1, lvl2 in column_names]
    
    # transform dtypes 
    for col in features:
        for col_dtype in INTERACTION_DTYPES:
            if col_dtype in col:
                features[col] = features[col].astype(INTERACTION_DTYPES[col_dtype])
    
    # add feature prefix to feature names
    cols_to_rename = [col for col in features.columns if col not in ['customer_id', 'article_id']]
    new_names = [feature_prefix + col for col in cols_to_rename]
    name_map = {old_name:new_name for old_name, new_name in zip(cols_to_rename, new_names)}
    features = features.rename(columns=name_map)
    
    return features


def build_interaction_features(
    data: pd.DataFrame,
    feat_start_date_: datetime, 
    feat_end_date_: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # select time range
    temp = data[(data['t_dat'] >= feat_start_date_) & 
                (data['t_dat'] <= feat_end_date_)].copy()
    # time decay in days and item_weight
    temp['days_diff'] = temp['t_dat'].apply(lambda x: (feat_end_date_ - x).days)
    temp['time_coef'] = 1/(1 + temp['days_diff'])
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    temp['price_weight'] = temp['time_coef'] * temp['price']
    logging.info('temp shape = %s', temp.shape)
    
    # user-item groupby features
    features_ui = temp.groupby(['customer_id', 'article_id'], as_index=False)\
                    .agg({'item_weight': {'sum', 'mean'}, 
                          'days_diff': {'min', 'mean'},
                          'price': 'sum',
                          'price_weight': {'mean', 'max', 'sum'},
                          'sales_channel_id': 'mean'})
    features_ui = features_process(features_ui, 'ui_')
    logging.info('features_ui shape = %s', features_ui.shape)  
    
    # user group by features
    features_u = temp.groupby(['customer_id'], as_index=False)\
                    .agg({'item_weight': {'min', 'mean', 'max', 'sum'}, 
                          'cnt_articles': {'mean', 'sum'}, 
                          't_dat': 'nunique',
                          'days_diff': {'min', 'mean'},
                          'price': {'min', 'mean', 'max', 'sum', 'std'},
                          'price_weight': {'min', 'mean', 'max', 'sum'},
                          'sales_channel_id': 'mean'})
    features_u = features_process(features_u, 'u_')
    logging.info('features_u shape = %s', features_u.shape)
    
    # item group by features
    features_i = temp.groupby(['article_id'], as_index=False)\
                    .agg({'item_weight': {'mean', 'sum'}, 
                          'cnt_articles': {'mean', 'sum'}, 
                          't_dat': 'nunique',
                          'days_diff': 'mean',
                          'price': {'min', 'mean', 'sum', 'std'},
                          'price_weight': {'min', 'mean', 'sum'},
                          'sales_channel_id': 'mean'})
    features_i = features_process(features_i, 'i_')
    logging.info('features_i shape = %s', features_i.shape)

    # user-item group by features for last week
    last_week_date = feat_end_date_ - timedelta(days=6) 
    features_uilw = temp[(temp['t_dat'] >= last_week_date)]\
                    .groupby(['customer_id', 'article_id'], as_index=False)\
                    .agg({'item_weight': {'sum', 'mean'},  
                          'days_diff': 'mean',
                          'price': 'sum',
                          'price_weight': {'mean', 'max', 'sum'},
                          'sales_channel_id': 'mean'})
    features_uilw = features_process(features_uilw, 'uilw_')
    logging.info('features_uilw shape = %s', features_uilw.shape)    

    # user-item group by features for last week
    features_ulw = temp[(temp['t_dat'] >= last_week_date)]\
                    .groupby(['customer_id'], as_index=False)\
                    .agg({'item_weight': {'min', 'mean', 'max', 'sum'}, 
                          'cnt_articles': {'mean', 'sum'}, 
                          'days_diff': {'min', 'mean'},
                          'price': {'min', 'mean', 'max', 'sum', 'std'},
                          'price_weight': {'min', 'mean', 'max', 'sum'},
                          'sales_channel_id': 'mean'})
    features_ulw = features_process(features_ulw, 'ulw_')
    logging.info('features_ulw shape = %s', features_ulw.shape) 

    # user-item group by features for last week
    features_ilw = temp[(temp['t_dat'] >= last_week_date)]\
                    .groupby(['article_id'], as_index=False)\
                    .agg({'item_weight': {'mean', 'sum'}, 
                          'cnt_articles': {'mean', 'sum'}, 
                          'days_diff': 'mean',
                          'price': {'min', 'mean', 'sum', 'std'},
                          'price_weight': {'min', 'mean', 'sum'},
                          'sales_channel_id': 'mean'})
    features_ilw = features_process(features_ilw, 'ilw_')
    logging.info('features_ilw shape = %s', features_ilw.shape) 
    
    return features_ui, features_u, features_i, \
        features_uilw, features_ulw, features_ilw


def get_features_2lvl(
    data: pd.DataFrame,
    feat_start_date_: datetime, 
    feat_end_date_: datetime,
    target_start_date_: datetime, 
    target_end_date_: datetime,    
    model_params: Dict[str, Any],
    customers_data_: pd.DataFrame,
    articles_data_: pd.DataFrame,
) -> pd.DataFrame:
    """Method for 2lvl features collection"""
    logging.info('get_features_2lvl STARTED')
    # get preds 1lvl
    preds_1lvl = implicit_fit_predict(
        data,
        feat_start_date_,
        feat_end_date_,
        model_params['params_1lvl']['user_item_values'],
        model_params['model_type_1lvl'],
        model_params['params_1lvl']['params'],
        model_params['params_1lvl']['similarity_type'],
        model_params['params_1lvl']['num_candidates_1lvl'],
        model_params['params_1lvl']["user_cnt_unq_items"],
        model_params['params_1lvl']["item_cnt_unq_users"],
        False
    )
    logging.info('len(preds_1lvl) = %s', len(preds_1lvl)) 
    
    # create user-item candidates from 1lvl
    candidates_df = build_candidates(preds_1lvl, model_params['params_1lvl']['use_item_rank'])
    logging.info('candidates_df shape = %s', candidates_df.shape)
    
    # start joining features
    features = candidates_df.copy()
    logging.info('features shape = %s', features.shape)
    
    # load target
    if target_start_date_ and target_end_date_:
        target = get_target(data, target_start_date_, target_end_date_)
        logging.info('len(target) = %s', len(target))
        target_df = get_target_df(target)
        logging.info('target_df shape = %s', target_df.shape)
    
        # merge target with 1s
        features = features.merge(target_df, on=['customer_id', 'article_id'], how='left')
        features['target'] = features['target'].fillna(0.).astype('uint8')
        logging.info('target merged, features shape = %s', features.shape)
        
        # compute 1lvl recall
        recall_df = features.groupby(['customer_id'], as_index=False).agg({'target':'sum'})
        recall_df['target_all'] = recall_df['customer_id'].apply(lambda x: len(target.get(x, [])))
        recall_df['user_recall'] = recall_df['target'] / recall_df['target_all']
        logging.info('Mean recall 1lvl = %s', recall_df['user_recall'].mean())
        
        # filter users with all zeros in target
        customers_to_use = recall_df[recall_df['target'] > 0]['customer_id'].tolist()
        features = features[features['customer_id'].isin(customers_to_use)].copy()
        logging.info('Customers with zero target filtered, features shape = %s', features.shape)
    
    # join user and item features
    features = features.merge(customers_data_, on=['customer_id'], how='left')
    logging.info('customers_data_ merged, features shape = %s', features.shape)
    features = features.merge(articles_data_, on=['article_id'], how='left')
    logging.info('articles_data_ merged, features shape = %s', features.shape)
    
    # build and join interaction features
    features_ui, features_u, features_i, \
        features_uilw, features_ulw, features_ilw = build_interaction_features(
            data,
            feat_start_date_, 
            feat_end_date_,
    )
    
    features = features.merge(features_ui, on=['customer_id', 'article_id'], how='left')
    logging.info('features_ui merged, features shape = %s', features.shape)
   
    features = features.merge(features_u, on=['customer_id'], how='left')
    logging.info('features_u merged, features shape = %s', features.shape)
    
    features = features.merge(features_i, on=['article_id'], how='left')
    logging.info('features_i merged, features shape = %s', features.shape)
    
    features = features.merge(features_uilw, on=['customer_id', 'article_id'], how='left')
    logging.info('features_uilw merged, features shape = %s', features.shape)
    
    features = features.merge(features_ulw, on=['customer_id'], how='left')
    logging.info('features_ulw merged, features shape = %s', features.shape)
    
    features = features.merge(features_ilw, on=['article_id'], how='left')
    logging.info('features_ilw merged, features shape = %s', features.shape)
    
    # logging.info('features.info() = %s', features.info())
    logging.info('get_features_2lvl DONE')
    return features
