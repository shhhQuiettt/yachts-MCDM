from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads dataset from csv file

    :param dataset_path: Path to dataset directory
    :return: Pandas Dataframe where every row represents single alternative, while every column represents single criterion
    """

    dataset = pd.read_csv(dataset_path / "dataset.csv", index_col=0)

    return dataset


def load_criterion_types(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    criterion_types = pd.read_csv(dataset_path / "type.csv", index_col=0)

    return criterion_types


def load_boundary_profiles(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    criterion_types = pd.read_csv(dataset_path / "boundary_profiles.csv", index_col=0)

    return criterion_types


def load_preference_thresholds(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    preference_thresholds = pd.read_csv(
        dataset_path / "preference_threshold.csv", index_col=0
    )

    return preference_thresholds


def load_indifference_thresholds(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    indifference_thresholds = pd.read_csv(
        dataset_path / "indifference_threshold.csv", index_col=0
    )

    return indifference_thresholds


def load_veto_thresholds(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    veto_thresholds = pd.read_csv(dataset_path / "veto_threshold.csv", index_col=0)

    return veto_thresholds


def load_credibility_threshold(dataset_path: Path) -> float:
    dataframe = pd.read_csv(dataset_path / "credibility_threshold.csv", index_col=0)

    return dataframe.iloc[0, 0]


def calculate_relation(
    relation_matrix: np.ndarray,
    alternatives: pd.Index,
    boundary_profiles_names: pd.Index,
) -> pd.DataFrame:
    print(relation_matrix.shape)

    return pd.DataFrame(
        relation_matrix, index=alternatives, columns=boundary_profiles_names
    )
