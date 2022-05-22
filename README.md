# Kaggle H&M Personalized Fashion Recommendations
Python code for 142nd place of
[Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/) 
competition.

## Description of competition
The task of this competition was to make product recommendations for every user in dataset 
for next week after available data.
There were data from previous transactions (2 years), customer data (1.3M) and item meta data (105k items), 
text and image data from product descriptions.

Main metric of competition is MAP@12. 

## Files
- `02_baselines_clean.ipynb` - baseline models validation (common, personal cold, frequent pairs, etc.)
- `04_implicit_knn_clean.ipynb` - implicit knn models validation (user2user, item2item)
- `05_2lvl_validation_clean.ipynb` - 2nd level model validation (implicit knn + LightGBM)
- `metrics.py` - MAP calculation (drop users without transactions in target period)
- `utils_category_top_items.py` - calculating top items in category
- `utils_features_2lvl.py` - features building and processing for 2lvl model
- `utils_frequent_pairs.py` - collection of frequent pairs of items
- `utils_heuristics.py` - common and personal cold start
- `utils_implicit_knn.py` - implicit knn recommender
- `utils_lightgbm.py` - lightgbm class
- `utils_models.py` - helpers for data processing for 1st level models
- `utils_validation.py` - 1st level validation
- `utils_validation_2lvl.py` - 2nd level validation

# Solution
Final solution (public MAP@12: 0.02530, private MAP@12: 0.02554) was an equal weight blend of:

1. *baseline combination* (public MAP@12: 0.02359):
   - personal history (30 days) with time decay (price weighted) 
   - frequent pairs for personal history less than 12 items 
   - gender personal cold start for users without items in 30 days period
2. *2nd level model* (public MAP@12: 0.02422):
   - implicit_knn: user2user, tfidf, user_item_value=item_weight (time decay), num_candidates=100 + age-gender personal cold start
   - LightGBM: num days in train = 30, user / item features, user-item / user / item transaction features for train period and for last week
3. *2nd level model* (public MAP@12: 0.02423):
   - implicit_knn: item2item, tfidf, user_item_value=item_weight (time decay), num_candidates=70 + common cold start - popular with time decay
   - LightGBM: num days in train = 30, user / item features, user-item / user / item transaction features for train period
4. *2nd level model* (public MAP@12: 0.02432):
   - implicit_knn: item2item, tfidf, user_item_value=price_weight (time decay), num_candidates=50 + age-gender personal cold start
   - LightGBM: num days in train = 30, user / item features, user-item / user / item transaction features for train period and for last week
5. *Best score public kernel* (public MAP@12: 0.0240)