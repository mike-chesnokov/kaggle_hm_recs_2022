from typing import List

import pandas as pd
from datetime import datetime


def popular_predict(data: pd.DataFrame, 
                    train_start_date_: datetime, 
                    train_end_date_: datetime,
                    num_candidates:int) -> List[int]:
    """
    make predictions of popular objects - cnt unique users
    """
    # compute top popular items on train
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)]\
        .groupby(['article_id'], as_index=False)\
        .agg({'customer_id': 'nunique'})\
        .sort_values('customer_id', ascending=False)
    return temp['article_id'][:num_candidates].tolist()


def popular_predict2(data: pd.DataFrame, 
                     train_start_date_: datetime,
                     train_end_date_: datetime,
                     num_candidates:int) -> List[int]:
    """
    make predictions of popular objects - cnt items
    """
    # compute top popular items on train
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)]\
        .groupby(['article_id'], as_index=False)\
        .agg({'cnt_articles': 'sum'})\
        .sort_values('cnt_articles', ascending=False)
    return temp['article_id'][:num_candidates].tolist()


def popular_time_decay_predict(
        data: pd.DataFrame,
        train_start_date_: datetime,
        train_end_date_: datetime,
        num_candidates:int,
        feature_sort:str = 'item_weight') -> List[int]:
    """
    make predictions of popular objects - time weighted items
    time decays in weeks - divide days by 7
    
    feature_sort: 'item_weight', 'price_weight', 'price'
    """
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    temp['price_weight'] = temp['time_coef'] * temp['price']
    
    # compute popular
    temp2 = temp\
        .groupby(['article_id'], as_index=False)\
        .agg({feature_sort: 'sum'})\
        .sort_values(feature_sort, ascending=False)
    return temp2['article_id'][:num_candidates].tolist()


def personal_popular_predict(data: pd.DataFrame,
                             train_start_date_: datetime, 
                             train_end_date_: datetime,
                             num_candidates:int) -> pd.DataFrame:
    """Get popular items from each customer (personal popular)
    return pd.DataFrame: "customer, articles_list"
    """
    # compute personal popular by max cnt_articles
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)]\
        .groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({'cnt_articles': 'sum'})\
        .sort_values(['customer_id','cnt_articles'], ascending=False)
    # get articles list
    temp2 = temp.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id': 'pred'})
    return temp2


def personal_recent_predict(data: pd.DataFrame,
                            train_start_date_: datetime, 
                            train_end_date_: datetime,
                            num_candidates:int) -> pd.DataFrame:
    """Get recent items from each customer (personal recent)
    return pd.DataFrame: "customer, articles_list"
    """
    # compute personal recent by max t_dat and max cnt_articles
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)]\
        .groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({'cnt_articles': 'sum', 't_dat':'max'})\
        .sort_values(['customer_id','t_dat', 'cnt_articles'], ascending=False)
    # get articles list
    temp2 = temp.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id':'pred'})
    temp2['pred'] = temp2['pred'].apply(lambda x: x[:num_candidates])
    return temp2


def personal_time_decay_predict(data: pd.DataFrame,
                                train_start_date_: datetime, 
                                train_end_date_: datetime,
                                num_candidates:int,
                                feature_sort:str = 'item_weight') -> pd.DataFrame:
    """Get popular items with time decay from each customer
    time decays in weeks - divide days by 7
    
    feature_sort: 'item_weight', 'price_weight', 'price'
    return pd.DataFrame: "customer, articles_list"
    """
    # compute personal recent by max t_dat and max cnt_articles
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    temp['price_weight'] = temp['time_coef'] * temp['price']
    
    # compute personal rank
    temp2 = temp\
        .groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({feature_sort: 'sum'})\
        .sort_values(['customer_id', feature_sort], ascending=False)
    
    # get articles list
    temp3 = temp2.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id': 'pred'})
    temp3['pred'] = temp3['pred'].apply(lambda x: x[:num_candidates])
    return temp3


def personal_history_time_decay_predict(
        data: pd.DataFrame,
        train_start_date_: datetime,
        train_end_date_: datetime,
        num_candidates: int) -> pd.DataFrame:
    """Get popular items with time decay from each customer
    return pd.DataFrame: "customer, articles_list"
    """
    # compute personal recent by max t_dat and max cnt_articles
    temp = data[(data['t_dat'] <= train_end_date_)].copy()
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    
    # compute personal rank
    temp2 = temp\
        .groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({'item_weight': 'sum'})\
        .sort_values(['customer_id', 'item_weight'], ascending=False)
    
    # get articles list
    temp3 = temp2.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id': 'pred'})
    temp3['pred'] = temp3['pred'].apply(lambda x: x[:num_candidates])
    return temp3


def personal_trending_time_decay_predict(
    data: pd.DataFrame,
    train_start_date_: datetime, 
    train_end_date_: datetime,
    num_candidates:int
) -> pd.DataFrame:
    """Get popular items with trending and time decay from each customer
    return pd.DataFrame: "customer, articles_list"
    """
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    # create number of week (from current to past)
    temp['week_num'] = temp['t_dat'].apply(lambda x: int((train_end_date_ - x).days/7))
    
    # calculate week sales and find sales for prev week
    week_sales = temp.groupby(['article_id', 'week_num'], as_index=False)\
                    .agg({'cnt_articles': 'sum'})\
                    .rename(columns={'cnt_articles': 'cnt_articles_week'})
    # create df with previous week number
    week_sales2 = week_sales.copy()
    week_sales2['week_num'] = week_sales2['week_num'].apply(lambda x: x - 1)
    # join prev and next week sales
    week_sales_3 = week_sales.merge(week_sales2, 
                                    how='left', 
                                    left_on=['article_id', 'week_num'], 
                                    right_on=['article_id', 'week_num'])
    week_sales_3['quotient'] = week_sales_3['cnt_articles_week_x']/week_sales_3['cnt_articles_week_y']
    week_sales_3['quotient'] = week_sales_3['quotient'].fillna(1.)
    
    # join quotient to original df
    temp = temp.merge(week_sales_3[['article_id', 'week_num', 'quotient']], 
                      on=['article_id', 'week_num'], 
                      how='left')
    temp['item_weight'] = temp['cnt_articles'] * temp['time_coef'] * temp['quotient']
    
    # compute personal rank
    temp2 = temp\
        .groupby(['customer_id', 'article_id'], as_index=False)\
        .agg({'item_weight': 'sum'})\
        .sort_values(['customer_id','item_weight'], ascending=False)

    # get articles list
    temp3 = temp2.groupby(['customer_id'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id': 'pred'})
    temp3['pred'] = temp3['pred'].apply(lambda x: x[:num_candidates])
    
    return temp3


def gender_age_personal_cold_start_predict(
    data: pd.DataFrame,
    train_start_date_: datetime, 
    train_end_date_: datetime,
    customers_data_: pd.DataFrame,
    articles_data_: pd.DataFrame,
    popular_articles: List[int],
    num_candidates: int,
    feature_sort: str = 'item_weight'
):
    """
    Get personal cold start based on age and gender
    """
    # compute recency by max t_dat and max cnt_articles
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    temp['price_weight'] = temp['time_coef'] * temp['price']
    
    # join features
    temp = temp.merge(articles_data_[['article_id', 'index_group_no']], how='left', on=['article_id'])
    temp = temp.merge(customers_data_[['customer_id', 'age_bin']], how='left', on=['customer_id'])
    
    temp2 = temp.groupby(['article_id', 'age_bin', 'index_group_no'], as_index=False)\
                .agg({feature_sort: 'sum'})\
                .sort_values([feature_sort], ascending=False)
        
    temp3 = temp2.groupby(['age_bin', 'index_group_no'], as_index=False)\
                 .agg({'article_id': list})\
                 .rename(columns={'article_id': 'pred'})
   
    # add popular preds for group without gender_calc (gender_calc=0)
    stub_df = [
        [18, 0, popular_articles], [25, 0, popular_articles], 
        [35, 0, popular_articles], [45, 0, popular_articles],
        [55, 0, popular_articles]
    ]

    temp_df = pd.DataFrame(stub_df, columns=['age_bin', 'index_group_no', 'pred'])
    temp3 = pd.concat([temp3, temp_df])
    
    # create a dict of category -> list of items
    temp3['pred'] = temp3['pred'].apply(lambda x: x[:num_candidates])
    category_top_items = {tuple((row['age_bin'], row['index_group_no'])): row['pred'] 
                          for row in temp3.to_dict(orient='records')}
    
    preds = customers_data_[['customer_id', 'age_bin', 'gender_calc']].copy()
    preds['personal_cold_start'] = preds[['age_bin','gender_calc']]\
        .apply(lambda x: category_top_items[(x[0], x[1])], axis=1)
    
    return preds[['customer_id', 'personal_cold_start']]
