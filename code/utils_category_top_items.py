from typing import Tuple, Dict, List

import pandas as pd
from datetime import datetime


def get_category_top_items_time_decay(
    data: pd.DataFrame,
    train_start_date_: datetime, 
    train_end_date_: datetime,
    articles_data_path: str,
    category_name: str,
    num_candidates:int
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Collect top items in category (field of articles_data) 
    by popularity weighted by time
    :return category_top_items: dict, category -> list of top items
    :return article_category: dict, article -> category  
    """
    # compute recency by max t_dat and max cnt_articles
    temp = data[(data['t_dat'] >= train_start_date_) & 
                (data['t_dat'] <= train_end_date_)].copy()
    # time decay in days
    temp['time_coef'] = temp['t_dat'].apply(lambda x: 1/(1 + (train_end_date_ - x).days))
    temp['item_weight'] = temp['time_coef'] * temp['cnt_articles']
    
    # calculate popularity ("item_weight") for each article
    temp2 = temp\
        .groupby(['article_id'], as_index=False)\
        .agg({'item_weight': 'sum'})\
        .sort_values(['item_weight'], ascending=False)
    
    # merge category to article
    articles = pd.read_feather(articles_data_path)
    article_category = {row['article_id']: row[category_name]
                        for row in articles[['article_id', category_name]].to_dict(orient='records')}
    temp2['category'] = temp2['article_id'].apply(lambda x: article_category[x])
    
    # collect popular items in each category
    temp3 = temp2.groupby(['category'], as_index=False)\
        .agg({'article_id': list})\
        .rename(columns={'article_id': 'pred'})

    temp3['pred'] = temp3['pred'].apply(lambda x: x[:num_candidates])
    category_top_items = {row['category']: row['pred']
                          for row in temp3.to_dict(orient='records')}
    
    return category_top_items, article_category


def concat_pred_cat_items(pred_: List[int],
                          article_category_: Dict[int, int],
                          category_top_items_: Dict[int, List[int]],
                          num_candidates: int) -> List[int]:
    """Join pred items and popular items from category"""
    pred_len = len(pred_)
    cnt_items_to_fill = num_candidates - pred_len
    items_to_fill = []

    for ind in range(cnt_items_to_fill):
        #  for cases of cnt items in pred_ < num_candidates/2
        # loop will go through pred_ list several times
        ind_ = ind%pred_len
        category = article_category_[pred_[ind_]]
        
        # check if any item exists
        if category in category_top_items_:
            category_items = category_top_items_[category]
            # iterate over category top
            for candidate in category_items:
                if candidate not in pred_ and candidate not in items_to_fill:
                    items_to_fill.append(candidate)
                    break
    return pred_ + items_to_fill


def category_top_items_predict(
        pred_: List[int],
        article_category_: Dict[int, int],
        category_top_items_: Dict[int, List[int]],
        num_candidates: int) -> List[int]:
    """Collect popular items from category and NOT JOIN THEM"""
    pred_len = len(pred_)
    items_to_fill = []

    for ind in range(num_candidates):
        #  for cases of cnt items in pred_ < num_candidates/2
        # loop will go through pred_ list several times
        ind_ = ind%pred_len
        category = article_category_[pred_[ind_]]
        
        # check if any item exists
        if category in category_top_items_:
            category_items = category_top_items_[category]
            # iterate over category top
            for candidate in category_items:
                if candidate not in pred_ and candidate not in items_to_fill:
                    items_to_fill.append(candidate)
                    break
    return items_to_fill    
