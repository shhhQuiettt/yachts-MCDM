from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils import (
    load_dataset,
    load_preference_information,
    display_ranking,
    Relation,
)


# TODO
def calculate_marginal_preference_index[T: (float, np.ndarray), U: (float, np.ndarray)](
    diff: T, q: U, p: U
) -> T:
    """
    Function that calculates the marginal preference index for the given pair of alternatives, according to the formula presented during the classes

    :param diff: difference between compared alternatives either as a float for single parser and alternative pairs, or as numpy array for multiple alternative/parser
    :param q: indifference threshold either as a float if you prefer to calculate for a single parser or as numpy array for multiple parser
    :param p: preference threshold either as a float if you prefer to calculate for a single parser or as numpy array for multiple parser
    :return: marginal preference index either as a float for single parser and alternative pairs, or as numpy array for multiple alternative/parser
    """
    if isinstance(diff, np.ndarray):
        t = np.where(diff <= q, 0, np.where(diff >= p, 1, (diff - q) / (p - q)))
        return t
    else:
        return 0 if diff <= q else 1 if diff >= p else (diff - q) / (p - q)
    


# TODO
def calculate_marginal_preference_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates the marginal preference matrix all alternatives pairs and criterion available in dataset

    :param dataset: difference between compared alternatives
    :param preference_information: preference information
    :return: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    """
    number_of_alternative = dataset.shape[0]
    number_of_criteria = preference_information.shape[0]
    
    marginal_preference_matrix = np.zeros((number_of_alternative, number_of_alternative, number_of_criteria))
    
    type_of_preference = preference_information['type']
    for i in range(number_of_alternative):
        for j in range(number_of_alternative):
            diff = np.where(type_of_preference == 'gain', dataset.iloc[i] - dataset.iloc[j], dataset.iloc[j] - dataset.iloc[i])

            marginal_preference_matrix[i, j] = calculate_marginal_preference_index(diff, preference_information.iloc[:, 0], preference_information.iloc[:, 1])
    
    return marginal_preference_matrix


# TODO
def calculate_comprehensive_preference_index(
    marginal_preference_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive preference index for the given dataset

    :param marginal_preference_matrix: 3D numpy array with marginal preference matrix on every parser, Consecutive indices [i, j, k] describe first alternative, second alternative, parser
    :param preference_information: Padnas preference information dataframe
    :return: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    """
    raise NotImplementedError()


# TODO
def calculate_positive_flow(
    comprehensive_preference_matrix: np.ndarray, index: pd.Index
) -> pd.Series:
    """
    Function that calculates the positive flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param index: index representing the alternative in the corresponding position in preference matrix
    :return: series representing positive flow values for the given preference matrix
    """
    raise NotImplementedError()


# TODO
def calculate_negative_flow(
    comprehensive_preference_matrix: np.ndarray, index: pd.Index
) -> pd.Series:
    """
    Function that calculates the negative flow value for the given preference matrix and corresponding index

    :param comprehensive_preference_matrix: 2D numpy array with marginal preference matrix. Every entry in the matrix [i, j] represents comprehensive preference index between alternative i and alternative j
    :param index: index representing the alternative in the corresponding position in preference matrix
    :return: series representing negative flow values for the given preference matrix
    """
    raise NotImplementedError()


# TODO
def calculate_net_flow(positive_flow: pd.Series, negative_flow: pd.Series) -> pd.Series:
    """
    Function that calculates the net flow value for the given positive and negative flow

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: series representing net flow values for the given preference matrix
    """
    raise NotImplementedError()


# TODO
def create_partial_ranking(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> set[tuple[str, str, Relation]]:
    """
    Function that aggregates positive and negative flow to a partial ranking (from Promethee I)

    :param positive_flow: series representing positive flow values for the given preference matrix
    :param negative_flow: series representing negative flow values for the given preference matrix
    :return: list of tuples when entries in a tuple represent first alternative, second alternative and the relation between them respectively
    """
    raise NotImplementedError()


# TODO
def create_complete_ranking(net_flow: pd.Series) -> set[tuple[str, str, Relation]]:
    """
    Function that aggregates positive and negative flow to a complete ranking (from Promethee II)
    :param net_flow: series representing net flow values for the given preference matrix
    :return: dataframe with alternatives in both index and columns. Every entry in the dataframe from row i and column j represents relation between alternative i and alternative j:
    1 means that i is preferred over j, or they are indifferent
    0 otherwise
    """
    raise NotImplementedError()


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def promethee(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    # print(dataset.head())
    preference_information = load_preference_information(dataset_path)

    # print(preference_information.head())
    marginal_preference_matrix = calculate_marginal_preference_matrix(
        dataset, preference_information
    )
    print(marginal_preference_matrix)
    # print(marginal_preference_matrix)
    # print(marginal_preference_matrix)
    # comprehensive_preference_matrix = calculate_comprehensive_preference_index(
    #     marginal_preference_matrix, preference_information
    # )

    # positive_flow = calculate_positive_flow(
    #     comprehensive_preference_matrix, dataset.index
    # )
    # negative_flow = calculate_negative_flow(
    #     comprehensive_preference_matrix, dataset.index
    # )

    # assert positive_flow.index.equals(negative_flow.index)

    # partial_ranking = create_partial_ranking(positive_flow, negative_flow)
    # display_ranking(partial_ranking, "Promethee I")

    # net_flow = calculate_net_flow(positive_flow, negative_flow)
    # complete_ranking = create_complete_ranking(net_flow)
    # display_ranking(complete_ranking, "Promethee II")


if __name__ == "__main__":
    promethee()
    # dataset = load_dataset(Path('data/lecture'))
    # preference_information = load_preference_information(Path('data/lecture'))
    # print(data.iloc[:,1])
    # diff = np.array([1, -200])
    # q = np.array([0.0, 100])
    # p = np.array([2, 300])
    # print(calculate_marginal_preference_index(diff, q, p))
    # t = np.array([[[1,2],[2,3]],[[1,2],[1,2]]])
    # print(t)
    # i = 0
    # j = 1
    # type_of_preference = preference_information['type']
    # diff = np.where(type_of_preference == 'gain', dataset.iloc[i] - dataset.iloc[j], dataset.iloc[j] - dataset.iloc[i])

    # t = calculate_marginal_preference_index(diff, preference_information.iloc[:, 1], preference_information.iloc[:, 2])

    # print(t)
