import pandas as pd
import numpy as np
import json
import os
import collections
from collections.abc import Set
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Dict, List
import warnings



# --------------------------------------------------------------------
# data processing

class OrderedSet(Set):
    def __init__(self, iterable=()):
        self.d = collections.OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


def shuffle_split_data(df, train_fraction, seed=0):
    df_train, df_test = train_test_split(df, train_size=train_fraction, random_state=seed, stratify=df["Target"])
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test

def shuffle_data(df, seed=0):
    df_ = shuffle(df, random_state=seed)
    df_.reset_index(inplace=True, drop=True)
    return df_


# plot histograms of the training data
def plot_hist(df, figsize, nrow, ncol):
    import matplotlib.pyplot as plt
    import seaborn as sns
    warnings.filterwarnings("ignore")
    plt.figure(figsize=figsize)
    for i, c in enumerate(df.columns):
        plt.subplot(nrow, ncol, i+1)
        sns.histplot(df[c])
        plt.title(c)
        plt.xlabel('')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


def scale_features(df_train, df_test, cols_list, scaler_list):
    df_train_scaled = pd.DataFrame(columns=df_train.columns)
    df_test_scaled = pd.DataFrame(columns=df_test.columns)
    scaler_params = {}

    for cols, scaler in zip(cols_list, scaler_list):
        if len(cols) == 0:
            continue

        cols_comp = list(OrderedSet(df_train.columns) - OrderedSet(cols))

        df_train_cols = df_train[cols]
        scaler.fit(df_train_cols)

        df_train_scaled[cols_comp] = df_train[cols_comp] if len(df_train_scaled)==0 else df_train_scaled[cols_comp]
        df_train_scaled[cols] = pd.DataFrame(scaler.transform(df_train_cols), columns=cols)

        df_test_scaled[cols_comp] = df_test[cols_comp] if len(df_test_scaled)==0 else df_test_scaled[cols_comp]
        df_test_scaled[cols] = pd.DataFrame(scaler.transform(df_test[cols]), columns=cols)

        for i, col in enumerate(cols):
            if 'MinMaxScaler' in str(type(scaler)):
                scaler_params[col] = {'min': scaler.data_min_[i], 'max': scaler.data_max_[i]}
            elif 'StandardScaler' in str(type(scaler)):
                scaler_params[col] = {'mean': scaler.mean_[i], 'std': scaler.scale_[i]}
            else:
                raise ValueError('Unimplemented scaler type: {}'.format(type(scaler)))

    if len(cols_list) == 0:
        df_train_scaled, df_test_scaled = df_train, df_test

    return df_train_scaled, df_test_scaled, scaler_params


def save_data(folderpath, con_features, cat_features, scaler_params, dtype_dict, df_train, df_test, df_val=None):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    df_train.to_csv(folderpath+'/train.csv', index=False, header=True)
    df_test.to_csv(folderpath+'/test.csv', index=False, header=True)
    if df_val is not None:
        df_val.to_csv(folderpath+'/val.csv', index=False, header=True)

    data_info = {
        "continuous_features": con_features,
        "categorical_features": cat_features,
        "scaler_params": scaler_params,
        "dtype_dict": {key: str(value) for key, value in dtype_dict.items()},
    }
    with open(folderpath+'/data_info.json', 'w') as fp:
        json.dump(data_info, fp)


def load_data(folderpath):
    df_train = pd.read_csv(folderpath+'/train.csv')
    df_test = pd.read_csv(folderpath+'/test.csv')
    df_val = pd.read_csv(folderpath+'/val.csv') if os.path.exists(folderpath+'/val.csv') else None

    X_train_df, y_train = df_train.drop("Target", axis=1), df_train["Target"]
    X_test_df, y_test = df_test.drop("Target", axis=1), df_test["Target"]
    X_val_df, y_val = None, None
    if df_val is not None:
        X_val_df, y_val = df_val.drop("Target", axis=1), df_val["Target"]

    infopath = folderpath + '/data_info.json'
    if os.path.isfile(infopath):
        with open(infopath, 'r') as fp:
            data_info = json.load(fp)

        continuous_features = data_info["continuous_features"]
        categorical_features = data_info["categorical_features"]
        scaler_params = data_info["scaler_params"]
        dtype_dict = data_info["dtype_dict"]

    else:
        continuous_features, categorical_features, scaler_params, dtype_dict = None, None, None, None

    return X_train_df, y_train, X_test_df, y_test, X_val_df, y_val, \
        continuous_features, categorical_features, scaler_params, dtype_dict



# --------------------------------------------------------------------
# tree-based target-aware feature discretization
# code adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
def compute_tree_bins(
        X: pd.DataFrame,
        y: pd.Series,
        con_feat: List[str],
        tree_kwargs: Dict[str, Any],
) -> List[np.ndarray]:
    """
    Args:
        X: The training features (DataFrame).
        y: The training labels (Series).
        con_feat: The list of continuous feature names.
        tree_kwargs: Keyword arguments for scikit-learn trees.
    Returns:
        A list of np array, where each array contains bin edges for one feature.
    """

    bins = []
    for col in con_feat:
        min_x, max_x = X[col].min(), X[col].max()
        if min_x == max_x:  # In case of same constant feature shuffled into the training set
            min_x, max_x = 0, 1
        feature_bin_edges = {min_x, max_x}

        tree_model = DecisionTreeClassifier(**tree_kwargs).fit(X[[col]], y)
        thresholds = tree_model.tree_.threshold
        is_branch_node = tree_model.tree_.children_left != tree_model.tree_.children_right
        feature_bin_edges.update(thresholds[is_branch_node])
        bins.append(np.sort(list(feature_bin_edges)))

    _check_bins(bins)
    return bins

def _check_bins(bins: List[np.ndarray]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for feature {i}: len(feature_bins)={len(feature_bins)}'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, for the {i}-th feature, the bin edges are not sorted'
            )
        if len(feature_bins) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
            )



