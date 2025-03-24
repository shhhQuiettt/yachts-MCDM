import enum
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads dataset from csv file

    :param dataset_path: Path to dataset directory
    :return: Pandas Dataframe where every row represents single alternative, while every column represents single criterion
    """

    dataset = pd.read_csv(dataset_path / "dataset.csv", index_col=0)

    return dataset


def load_boundary_profiles(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads boundary profiles from csv file

    :param dataset_path: Path to dataset directory
    :return: Pandas Dataframe where every row represents single boundary profile, while every column represents single criterion
    """

    boundary_profiles = pd.read_csv(dataset_path / "boundary_profiles.csv", index_col=0)

    return boundary_profiles


def load_preference_information(dataset_path: Path) -> pd.DataFrame:
    """
    Function that loads preference information from csv file

    :param dataset_path: Path to dataset directory
    :return:
    """
    preferences = pd.read_csv(dataset_path / "preference.csv", index_col=0)

    return preferences


class Relation(enum.Flag):
    INDIFFERENT = enum.auto()
    PREFERRED = enum.auto()
    INCOMPARABLE = enum.auto()


def convert_to_outranking_matrix(
    ranking: set[tuple[str, str, Relation]],
) -> pd.DataFrame:
    index = list({y for x in ranking for y in x[:2]})

    result = pd.DataFrame(
        np.zeros((len(index),) * 2), index=index, columns=index, dtype=np.int64
    )

    for first, second, relation in ranking:
        match relation:
            case Relation.INDIFFERENT:
                result.loc[first, second] = result.loc[first, second] = 1
            case Relation.PREFERRED:
                result.loc[first, second] = 1

    return result


def merge_nodes(ranking: pd.DataFrame, first_index, second_index) -> None:
    ranking = ranking.copy()

    index_ = ranking.index.to_list()
    index_[first_index] = f"{ranking.index[first_index]}, {ranking.index[second_index]}"
    ranking.index = index_
    ranking.columns = index_

    ranking.drop(labels=ranking.index[second_index], inplace=True)
    ranking.drop(labels=ranking.index[second_index], axis=1, inplace=True)


def remove_cycles(ranking: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()

    while True:
        cycles = np.triu(ranking & ranking.T, k=-1)
        indices = np.stack(np.nonzero(cycles), axis=1)

        if indices.size == 0:
            break

        first, second = indices[0].tolist()
        merge_nodes(ranking, first, second)

    return ranking


def display_ranking(ranking: set[tuple[str, str, Relation]], title: str) -> None:
    outranking_matrix = convert_to_outranking_matrix(ranking)

    diagram = remove_cycles(outranking_matrix)

    nodes = diagram.index.tolist()
    edges = [(nodes[i], nodes[j]) for i, j in np.stack(np.nonzero(diagram)).T.tolist()]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g = nx.transitive_reduction(g)

    layout = graphviz_layout(g, prog="dot")
    plt.title(title)
    nx.draw(g, layout, with_labels=True, arrows=True)
    plt.show()
