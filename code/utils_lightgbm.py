from typing import Dict, Any

import copy
import shap
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt


class LightGBMRecommender:
    """
    Class for second level LightGBM model
    """
    TARGET_NAME = 'target'

    def __init__(
            self,
            num_trees: int = 100,
            early_stopping_rounds: int = 30,
            verbose_eval: int = 10,
            **model_params: Dict[str, Any]
    ):
        self.num_trees = num_trees
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model_params = model_params
        self.model = None
        self.eval_result = {}

    def fit(self,
            train_features: pd.DataFrame,
            valid_features: pd.DataFrame = None
            ):
        valid_sets = []
        valid_names = []
        lgb_train = lgb.Dataset(
            train_features.drop(columns=[self.TARGET_NAME]),
            label=train_features[self.TARGET_NAME],
            categorical_feature='auto',  # infer pandas 'category' data type
            free_raw_data=True
        )
        valid_sets.append(lgb_train)
        valid_names.append('train')

        if valid_features is not None:
            lgb_valid = lgb.Dataset(
                valid_features.drop(columns=[self.TARGET_NAME]),
                label=valid_features[self.TARGET_NAME],
                categorical_feature='auto',  # infer pandas 'category' data type
                free_raw_data=True
            )
            valid_sets.append(lgb_valid)
            valid_names.append('valid')
        # train model
        self.model = lgb.train(
            self.model_params,
            train_set=lgb_train,
            num_boost_round=self.num_trees,
            valid_sets=valid_sets,  # 2 datasets: train and valid
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, first_metric_only=True),
                       lgb.log_evaluation(period=self.verbose_eval),
                       lgb.record_evaluation(self.eval_result)]
        )

    def predict(
            self,
            features: pd.DataFrame,
            num_candidates: int = 50,
            return_plain_df: bool = False
    ):
        """
        if "return_plain_df = True" method return dataframe with sorted items by score
        else return dict -> customer_id -> List[article_id]
        """
        if self.TARGET_NAME in features.columns:
            scores = self.model.predict(features.drop(columns=[self.TARGET_NAME]))
        else:
            scores = self.model.predict(features)
            
        sorted_scores = features.copy()
        sorted_scores['score'] = scores
        if return_plain_df:
            sorted_scores = sorted_scores \
                .sort_values(['customer_id', 'score'],
                             kind='stable',
                             ascending=False).copy()
            return sorted_scores
        else:
            sorted_scores = sorted_scores[['customer_id', 'article_id', 'score']] \
                .sort_values(['customer_id', 'score'],
                             kind='stable',
                             ascending=False).copy()
            predictions = sorted_scores\
                .groupby(['customer_id'], as_index=False)\
                .agg({'article_id': list})
            predictions['article_id'] = predictions['article_id'].apply(lambda x: x[:num_candidates])
            predictions = predictions.rename(columns={'article_id':'pred'})
            #predictions_dct = {row['customer_id']: row['article_id']
            #                   for row in predictions.to_dict(orient='records')}
            #return predictions_dct
            return predictions

    def plot_feature_importance(self, max_features=40):
        """
        Method to plot LightGBM feature importance
        :param  max_features: number of features to plot
        """
        fig, axis = plt.subplots(figsize=(12, 12))
        lgb.plot_importance(self.model,
                            max_num_features=max_features,
                            height=0.8,
                            ax=axis,
                            importance_type='gain')
        axis.grid(False)
        plt.tight_layout()
        plt.title("LightGBM - Feature Importance", fontsize=20)
        return fig

    def plot_train_metric(self, metric='auc'):
        """
        Method to plot LightGBM train metrics
        :param  metric: number of features to plot
        """
        fig, axis = plt.subplots(figsize=(12, 5))
        lgb.plot_metric(self.eval_result,
                        metric=metric,
                        ax=axis)
        axis.grid(which='major', linestyle='--', linewidth=1, alpha=0.3)
        plt.tight_layout()
        plt.title("LightGBM - train metrics", fontsize=20)
        return fig
    
    def calculate_shap_values(self, features_):
        """
        Method for calculating shap values
        """
        if self.TARGET_NAME in features_.columns:
            features_ = features_.drop(columns=[self.TARGET_NAME])
        explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(features_)
        shap_obj = explainer(features_)

        # save values only for target = 1
        # values calculated for 2 classes
        shap_obj2 = copy.deepcopy(shap_obj)
        shap_obj2.values = shap_obj2.values[:, :, 1]
        shap_obj2.base_values = shap_obj2.base_values[:, 1]
        return shap_obj2

    @staticmethod
    def plot_shap_values(shap_obj, plot_type='bar', max_features=50):
        """
        Method to plot shap summary plot
        plot_type: str, possible values - 'bar', 'beeswarm'
        """
        fig = plt.figure()
        if plot_type == 'bar':
            shap.plots.bar(shap_obj, max_display=max_features, show=False)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_obj, max_display=max_features, show=False)
        title = f'SHAP Values, plot_type={plot_type}'
        plt.title(title, fontsize=20)
        plt.tight_layout()
        return fig
