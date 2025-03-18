import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_utils import load_data, shuffle_split_data, compute_tree_bins



@dataclass
class DataConfig:
    dataset: str
    seed: int
    batch_size: int
    continuous_resolution: int = 4
    discretize_strategy: str = 'tree'


class TabularDataset:
    def __init__(self, config: DataConfig):
        self.config = config
        self.datapath = get_datapath(config.dataset, config.seed)
        self.input_dim = None
        self.num_con_feat = None
        self.num_cat_feat = None
        self.num_classes = None
        self.num_train_samples, self.num_test_samples = None, None
        self.inverse_class_weights = None
        self.feat_names = None
        self.scaler_params = None
        self.dtype_dict = None
        self.bins_thresholds = None  # store the thresholds for discretization
        self.get_attributes()


    def get_attributes(self):
        (X_train_df, y_train, X_test_df, y_test, X_val_df, y_val,
         con_feat, cat_feat, scaler_params, dtype_dict) = load_data(self.datapath)

        if len(con_feat) == 0 or self.config.continuous_resolution < 2:
            self.input_dim = len(cat_feat) + len(con_feat)

        else:  # bin-based discretization
            n_bins_per_feat = self.config.continuous_resolution + 1  # only takes the non-terminal bins cutoffs later
            if self.config.discretize_strategy == 'tree':
                bin_edges = compute_tree_bins(
                    X=X_train_df,
                    y=y_train,
                    con_feat=con_feat,
                    tree_kwargs={'max_leaf_nodes':n_bins_per_feat,
                                 'random_state':self.config.seed,
                                 'class_weight':'balanced'},
                )
            elif self.config.discretize_strategy in ['uniform', 'quantile', 'kmeans']:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins_per_feat, encode='ordinal',
                    strategy=self.config.discretize_strategy, random_state=self.config.seed)
                discretizer.fit(X_train_df[con_feat])
                bin_edges = discretizer.bin_edges_
            else:
                raise ValueError('Invalid discretize_strategy {}'.format(self.config.discretize_strategy))

            self.bins_thresholds = [np.array(bin_edge[1:-1]) for bin_edge in bin_edges]
            self.input_dim = len(cat_feat) + sum(len(bin_th) for bin_th in self.bins_thresholds)

        self.num_con_feat = len(con_feat)
        self.num_cat_feat = len(cat_feat)
        self.num_classes = len(np.unique(y_train))
        self.num_train_samples, self.num_test_samples = len(y_train), len(y_test)

        class_counts = np.bincount(y_train, minlength=self.num_classes)
        self.inverse_class_weights = self.num_train_samples / self.num_classes / (class_counts + 0.1)

        # process feature names and other info for network visualization
        self.process_feat_info(con_feat, scaler_params, dtype_dict)


    def get_data(self):
        X_train_df, y_train, X_test_df, y_test, _, _, con_feat, _, _, _ = load_data(self.datapath)

        if self.num_con_feat > 0 and self.config.continuous_resolution >= 2:
            assert self.bins_thresholds is not None
            repeats = [len(l) for l in self.bins_thresholds]
            X_train_df = self.repeat_feat(X_train_df, con_feat, repeats)
            X_test_df = self.repeat_feat(X_test_df, con_feat, repeats)

        return X_train_df, y_train, X_test_df, y_test


    def get_dataLoader(self, num_workers, isEval=False, val_split=None, split_seed=None):
        # isEval: whether in evaluation mode (no shuffle, no drop_last)
        X_train_df, y_train, X_test_df, y_test = self.get_data()

        validation_loader = None
        if val_split is not None:
            assert split_seed is not None
            df_train, df_val = shuffle_split_data(
                pd.concat([X_train_df, y_train.to_frame()], axis=1), val_split, seed=split_seed)
            X_train_df, y_train = df_train.drop("Target", axis=1), df_train["Target"]
            X_val_df, y_val = df_val.drop("Target", axis=1), df_val["Target"]
            validation_loader = get_loader(X_val_df, y_val, False, 1024, num_workers)

        train_loader = get_loader(X_train_df, y_train, not isEval, self.config.batch_size, num_workers)
        test_loader = get_loader(X_test_df, y_test, False, 1024, num_workers)
        return train_loader, validation_loader, test_loader


    def get_cv_dataLoader(self, cvn, num_workers, train_val_split=0.8):
        X_train_df, y_train, _, _ = self.get_data()
        if cvn == 1:  # no cross validation, just use train_val_split to do train/val split
            df_train, df_val = shuffle_split_data(
                pd.concat([X_train_df, y_train.to_frame()], axis=1), train_val_split, seed=self.config.seed)
            X_train, y_train = df_train.drop("Target", axis=1), df_train["Target"]
            X_val, y_val = df_val.drop("Target", axis=1), df_val["Target"]
            train_loader = get_loader(X_train, y_train, True, self.config.batch_size, num_workers)
            val_loader = get_loader(X_val, y_val, False, 1024, num_workers)
            yield train_loader, val_loader

        else:
            kf = StratifiedKFold(n_splits=cvn, shuffle=True, random_state=self.config.seed)
            for train_index, val_index in kf.split(X_train_df, y_train):
                X_train_fold, y_train_fold = X_train_df.iloc[train_index], y_train.iloc[train_index]
                X_val_fold, y_val_fold = X_train_df.iloc[val_index], y_train.iloc[val_index]
                train_loader = get_loader(X_train_fold, y_train_fold, True, self.config.batch_size, num_workers)
                val_loader = get_loader(X_val_fold, y_val_fold, False, 1024, num_workers)
                yield train_loader, val_loader


    @staticmethod
    def repeat_feat(df, con_feat, repeats):
        if len(con_feat) == 0:
            return df
        repeated_cols = {}
        for idx, col in enumerate(con_feat):
            for r in range(repeats[idx]):
                repeated_cols[f'{col}_{r}'] = df[col].values
        repeated_cols_df = pd.DataFrame(repeated_cols)
        return pd.concat([
            df.drop(con_feat, axis=1).reset_index(drop=True),
            repeated_cols_df.reset_index(drop=True)], axis=1)


    def get_class_weights(self, balance_class_weights=True):
        if balance_class_weights:
            return torch.Tensor(self.inverse_class_weights)
        return None


    def get_threshold_info(self):
        if self.bins_thresholds is not None:
            threshold_init = np.concatenate(self.bins_thresholds)
        else:
            threshold_init = np.full(self.num_con_feat, 0.5)
        return len(threshold_init), threshold_init


    # feat_names, scaler_params, and dtype_dict are used for network visualization
    def process_feat_info(self, con_feat, scaler_params, dtype_dict):
        def process_name(feat_name):
            feat_name = feat_name.replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace(')', '_')
            feat_name = re.sub(r'_+', '_', feat_name)
            feat_name = feat_name.rstrip('_')
            return feat_name

        X_test_df_processed = self.get_data()[2]
        feat_names_ = X_test_df_processed.columns.to_list()
        feat_names = []
        for name in feat_names_:
            temp = re.sub(r'_\d+$', '', name)  # check for repeated features
            feat_names.append(temp if temp in con_feat else name)

        self.feat_names = [process_name(name) for name in feat_names]
        self.scaler_params = {process_name(key): value for key, value in scaler_params.items()}
        self.dtype_dict = {process_name(key): value for key, value in dtype_dict.items()}


    def __repr__(self):
        return f"DataConfig: {self.config}"



def get_datapath(dataset, seed):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    abs_datapath = os.path.join(dir_path, f'datasets/{dataset}/seed_{seed}/data')
    return abs_datapath


def get_loader(X, y, isTrain, batch_size, num_workers):
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    data = TensorDataset(torch.from_numpy(X).to(dtype=torch.float32), torch.from_numpy(y).long())
    return DataLoader(data, batch_size=batch_size,
                      shuffle=isTrain, drop_last=isTrain,
                      num_workers=num_workers,)


# output data stats, used by Ray Tune
def get_data_stats_tune(dataset_name, seed=0, continuous_resolution=4, discretize_strategy='uniform'):
    _, y_train, _, _, _, _, con_feat, cat_feat, _, _ = load_data(get_datapath(dataset_name, seed))
    # for simplicity, use default values to calculate input dim because tune.grid_search
    # is not able to use continuous_resolution and discretize_strategy from config
    input_dim = len(cat_feat) + continuous_resolution * len(con_feat)
    num_classes = len(np.unique(y_train))
    num_train_samples = len(y_train)
    return input_dim, num_classes, num_train_samples


