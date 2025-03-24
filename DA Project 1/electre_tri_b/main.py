from pathlib import Path

import click
import numpy as np
import pandas as pd

from electre_tri_b.utils import (
    load_dataset,
    load_boundary_profiles,
    load_indifference_thresholds,
    load_preference_thresholds,
    load_veto_thresholds,
    load_criterion_types,
    load_credibility_threshold,
)


# TODO
def calculate_marginal_concordance_index[
    T: (float, np.ndarray),
    U: (float, np.ndarray),
](diff: T, q: U, p: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param q: indifference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal concordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    raise NotImplementedError()


# TODO
def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    indifference_thresholds,
    preference_thresholds,
    criterion_types,
) -> np.ndarray:
    """
    Function that calculates the marginal concordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param indifference_thresholds: pandas dataframe representing indifference thresholds for all boundary profiles and criterion
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    """
    raise NotImplementedError()


# TODO
def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray, criterion_types: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive concordance matrix for the given dataset

    :param marginal_concordance_matrix: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    :param criterion_types: dataframe that contains "k" column with criterion weights
    :return: 3D numpy array with comprehensive concordance matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe comprehensive concordance index between alternative i and boundary profile j, while element with index [1, i, j] describe comprehensive concordance index between boundary profile j and  alternative i
    """
    raise NotImplementedError()


# TODO
def calculate_marginal_discordance_index[
    T: (float, np.ndarray),
    U: (float, np.ndarray),
](diff: T, p: U, v: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param v: veto threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal discordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    raise NotImplementedError()


# TODO
def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    preference_thresholds,
    veto_thresholds,
    criterion_types,
) -> np.ndarray:
    """
    Function that calculates the marginal discordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param veto_thresholds: pandas dataframe representing veto thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal discordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal discordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal discordance index between boundary profile j and  alternative i on criterion k
    """
    raise NotImplementedError()


# TODO
def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Function that calculates the credibility index for the given comprehensive concordance matrix and marginal discordance matrix

    :param comprehensive_concordance_matrix: 3D numpy array with comprehensive concordance matrix. Every entry in the matrix [i, j] represents comprehensive concordance index between alternative i and alternative j
    :param marginal_discordance_matrix: 3D numpy array with marginal discordance matrix, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :return: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    """
    raise NotImplementedError()


# TODO
def calculate_outranking_relation_matrix(
    credibility_index: np.ndarray, credibility_threshold
) -> np.ndarray:
    """
    Function that calculates boolean matrix with information if outranking holds for a given pair

    :param credibility_index: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    :param credibility_threshold: float number
    :return: 3D numpy boolean matrix with information if outranking holds for a given pair
    """
    raise NotImplementedError()


# TODO
def calculate_relation(
    outranking_relation_matrix: np.ndarray,
    alternatives: pd.Index,
    boundary_profiles_names: pd.Index,
) -> pd.DataFrame:
    """
    Function that determine relation between alternatives and boundary profiles

    :param outranking_relation_matrix: 3D numpy boolean matrix with information if outranking holds for a given pair
    :param alternatives: names of alternatives
    :param boundary_profiles_names: names of boundary profiles
    :return: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. Use "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    """
    raise NotImplementedError()


# TODO
def calculate_pessimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates pessimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with pessimistic assigment
    """
    raise NotImplementedError()


# TODO
def calculate_optimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates optimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with optimistic assigment
    """
    raise NotImplementedError()


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def electre(dataset_path: Path) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    boundary_profiles = load_boundary_profiles(dataset_path)
    criterion_types = load_criterion_types(dataset_path)
    indifference_thresholds = load_indifference_thresholds(dataset_path)
    preference_thresholds = load_preference_thresholds(dataset_path)
    veto_thresholds = load_veto_thresholds(dataset_path)
    credibility_threshold = load_credibility_threshold(dataset_path)

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(
        dataset,
        boundary_profiles,
        indifference_thresholds,
        preference_thresholds,
        criterion_types,
    )
    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(
        marginal_concordance_matrix, criterion_types
    )

    marginal_discordance_matrix = calculate_marginal_discordance_matrix(
        dataset,
        boundary_profiles,
        preference_thresholds,
        veto_thresholds,
        criterion_types,
    )

    credibility_index = calculate_credibility_index(
        comprehensive_concordance_matrix, marginal_discordance_matrix
    )
    outranking_relation_matrix = calculate_outranking_relation_matrix(
        credibility_index, credibility_threshold
    )
    relation = calculate_relation(
        outranking_relation_matrix, dataset.index, boundary_profiles.index
    )

    pessimistic_assigment = calculate_pessimistic_assigment(relation)
    optimistic_assigment = calculate_optimistic_assigment(relation)

    print("pessimistic assigment\n", pessimistic_assigment)
    print("optimistic assigment\n", optimistic_assigment)


if __name__ == "__main__":
    electre()
