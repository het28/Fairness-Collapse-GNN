"""
Multigroup preprocessing: Setting A (binary Y + multigroup S) and Setting B (multiclass Y + multigroup S).
Credit, German, Bail only. Uses same graph structure and _create_masks, _remap_edges from preprocess_data.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from preprocess_data import _create_masks, _remap_edges


# ---------- Setting A: binary Y + multigroup S ----------

def _load_credit_setting_a(data_dir):
    """Credit Setting A: Y=NoDefaultNextMonth (binary), S=EducationLevel (0-3)."""
    df_path = os.path.join(data_dir, "credit.csv")
    edges_path = os.path.join(data_dir, "credit_edges.csv")
    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, "user_id", user_labels.index)

    sens_attr = torch.tensor(user_labels["EducationLevel"].values, dtype=torch.long)
    labels = torch.tensor(user_labels["NoDefaultNextMonth"].values, dtype=torch.long)
    node_features_df = user_labels.drop(columns=["user_id", "NoDefaultNextMonth", "EducationLevel"])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 2
    return data, sens_attr


def _load_german_setting_a(data_dir):
    """German Setting A: Y=GoodCustomer (binary), S=AgeGroup quartiles [19-27], (27-33], (33-42], (42-75]."""
    df_path = os.path.join(data_dir, "german.csv")
    edges_path = os.path.join(data_dir, "german_edges.csv")
    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, "user_id", user_labels.index)
    user_labels["Gender"] = user_labels["Gender"].replace({"Female": 1, "Male": 0})
    user_labels["GoodCustomer"] = user_labels["GoodCustomer"].replace({1: 1, -1: 0})

    age = user_labels["Age"].values
    q25, q50, q75 = np.percentile(age, [25, 50, 75])
    bins = [-np.inf, q25, q50, q75, np.inf]
    age_group = np.digitize(age, bins) - 1  # 0..3
    age_group = np.clip(age_group, 0, 3)

    sens_attr = torch.tensor(age_group, dtype=torch.long)
    labels = torch.tensor(user_labels["GoodCustomer"].values, dtype=torch.long)
    node_features_df = user_labels.drop(columns=["user_id", "GoodCustomer", "Gender", "PurposeOfLoan", "Age"])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 2
    return data, sens_attr


def _load_bail_setting_a(data_dir):
    """Bail Setting A: Y=RECID (binary), S=AgeGroup quartiles on AGE."""
    df_path = os.path.join(data_dir, "bail.csv")
    edges_path_csv = os.path.join(data_dir, "bail_edges.csv")
    edges_path_txt = os.path.join(data_dir, "bail_edges.txt")
    user_labels = pd.read_csv(df_path)
    if os.path.isfile(edges_path_csv):
        user_edges = pd.read_csv(edges_path_csv)
    else:
        user_edges = pd.read_csv(edges_path_txt, sep=r"\s+", header=None, names=["uid1", "uid2"])
    user_labels.insert(0, "user_id", user_labels.index)

    age = user_labels["AGE"].values
    q25, q50, q75 = np.percentile(age, [25, 50, 75])
    bins = [-np.inf, q25, q50, q75, np.inf]
    age_group = np.digitize(age, bins) - 1
    age_group = np.clip(age_group, 0, 3)

    sens_attr = torch.tensor(age_group, dtype=torch.long)
    labels = torch.tensor(user_labels["RECID"].values, dtype=torch.long)
    node_features_df = user_labels.drop(columns=["user_id", "RECID", "WHITE", "AGE"])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 2
    return data, sens_attr


# ---------- Setting B: multiclass Y + multigroup S ----------

def _load_credit_setting_b(data_dir):
    """Credit Setting B: Y=TotalMonthsOverdue binned (0, 1-2, 3-5, 6+), S=EducationLevel (0-3)."""
    df_path = os.path.join(data_dir, "credit.csv")
    edges_path = os.path.join(data_dir, "credit_edges.csv")
    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, "user_id", user_labels.index)

    # Y: 4 classes from TotalMonthsOverdue: 0 -> 0, 1-2 -> 1, 3-5 -> 2, 6+ -> 3
    tmo = user_labels["TotalMonthsOverdue"].values
    y_bin = np.zeros(len(tmo), dtype=np.int64)
    y_bin[(tmo >= 1) & (tmo <= 2)] = 1
    y_bin[(tmo >= 3) & (tmo <= 5)] = 2
    y_bin[tmo >= 6] = 3

    sens_attr = torch.tensor(user_labels["EducationLevel"].values, dtype=torch.long)
    labels = torch.tensor(y_bin, dtype=torch.long)
    node_features_df = user_labels.drop(
        columns=["user_id", "NoDefaultNextMonth", "EducationLevel", "TotalMonthsOverdue"]
    )
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 4
    return data, sens_attr


def _load_german_setting_b(data_dir):
    """German Setting B: Y=LoanAmount tertiles (3 classes), S=AgeGroup quartiles."""
    df_path = os.path.join(data_dir, "german.csv")
    edges_path = os.path.join(data_dir, "german_edges.csv")
    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, "user_id", user_labels.index)
    user_labels["Gender"] = user_labels["Gender"].replace({"Female": 1, "Male": 0})

    age = user_labels["Age"].values
    q25, q50, q75 = np.percentile(age, [25, 50, 75])
    bins = [-np.inf, q25, q50, q75, np.inf]
    age_group = np.digitize(age, bins) - 1
    age_group = np.clip(age_group, 0, 3)

    # Y: 3 classes from LoanAmount tertiles
    la = user_labels["LoanAmount"].values
    t33, t66 = np.percentile(la, [33.33, 66.67])
    y_bin = np.zeros(len(la), dtype=np.int64)
    y_bin[la > t33] = 1
    y_bin[la > t66] = 2

    sens_attr = torch.tensor(age_group, dtype=torch.long)
    labels = torch.tensor(y_bin, dtype=torch.long)
    node_features_df = user_labels.drop(
        columns=["user_id", "GoodCustomer", "Gender", "PurposeOfLoan", "Age", "LoanAmount"]
    )
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 3
    return data, sens_attr


def _load_bail_setting_b(data_dir):
    """Bail Setting B: Y=PRIORS binned (0, 1-2, 3-5, 6+), S=AgeGroup quartiles on AGE."""
    df_path = os.path.join(data_dir, "bail.csv")
    edges_path_csv = os.path.join(data_dir, "bail_edges.csv")
    edges_path_txt = os.path.join(data_dir, "bail_edges.txt")
    user_labels = pd.read_csv(df_path)
    if os.path.isfile(edges_path_csv):
        user_edges = pd.read_csv(edges_path_csv)
    else:
        user_edges = pd.read_csv(edges_path_txt, sep=r"\s+", header=None, names=["uid1", "uid2"])
    user_labels.insert(0, "user_id", user_labels.index)

    age = user_labels["AGE"].values
    q25, q50, q75 = np.percentile(age, [25, 50, 75])
    bins = [-np.inf, q25, q50, q75, np.inf]
    age_group = np.digitize(age, bins) - 1
    age_group = np.clip(age_group, 0, 3)

    # Y: 4 classes from PRIORS: 0 -> 0, 1-2 -> 1, 3-5 -> 2, 6+ -> 3
    priors = user_labels["PRIORS"].values
    y_bin = np.zeros(len(priors), dtype=np.int64)
    y_bin[(priors >= 1) & (priors <= 2)] = 1
    y_bin[(priors >= 3) & (priors <= 5)] = 2
    y_bin[priors >= 6] = 3

    sens_attr = torch.tensor(age_group, dtype=torch.long)
    labels = torch.tensor(y_bin, dtype=torch.long)
    node_features_df = user_labels.drop(columns=["user_id", "RECID", "WHITE", "AGE", "PRIORS"])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    edge_index = _remap_edges(user_labels, user_edges)
    data = Data(x=node_features, edge_index=edge_index, y=labels)
    data.num_classes = 4
    return data, sens_attr


# ---------- Dispatcher ----------

_LOADERS_A = {
    "credit": _load_credit_setting_a,
    "german": _load_german_setting_a,
    "bail": _load_bail_setting_a,
}

_LOADERS_B = {
    "credit": _load_credit_setting_b,
    "german": _load_german_setting_b,
    "bail": _load_bail_setting_b,
}


def preprocess_multigroup(
    data_dir,
    data_name,
    setting,
    train_split=0.8,
    test_split=0.1,
    split_seed=42,
    stratify_by=None,
):
    """
    Load dataset for multigroup setting A or B.
    setting: "a" | "b"
    stratify_by: None | "sens_label" (stratify by S and Y for balanced splits).
    Returns: (data, sens_attr). data has .num_classes (2 for A, 3 or 4 for B).
    """
    if setting not in ("a", "b"):
        raise ValueError("setting must be 'a' or 'b'")
    loaders = _LOADERS_A if setting == "a" else _LOADERS_B
    if data_name not in loaders:
        raise ValueError(f"data_name must be one of {list(loaders.keys())}")

    dataset_path = os.path.join(data_dir, data_name)
    data, sens_attr = loaders[data_name](dataset_path)

    stratify_labels = None
    if stratify_by == "sens_label":
        sens = sens_attr.numpy().ravel()
        y = data.y.numpy().ravel()
        n_s = int(sens.max()) + 1
        n_y = getattr(data, "num_classes", 2)
        stratify_labels = sens * n_y + y  # unique stratum per (S, Y)

    data.train_mask, data.val_mask, data.test_mask = _create_masks(
        data.num_nodes,
        train_split,
        test_split,
        seed=split_seed,
        stratify_labels=stratify_labels,
    )
    print(f"Multigroup setting {setting.upper()}: loaded '{data_name}', num_classes={data.num_classes}.")
    return data, sens_attr
