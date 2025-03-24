import numpy.typing as npt
import pandas as pd

CriterionType = str


def calculate_marginal_preference_index(): ...


def calculate_marginal_preference_matrix(
    data: npt.NDArray,
) -> dict[CriterionType, npt.NDArray]: ...


def calculate_comprehensive_preference_index(
    marginal_preferences: dict[CriterionType, npt.NDArray],
) -> npt.NDArray: ...


def calculate_positive_flow(
    comprehensive_preference_index: npt.NDArray,
) -> npt.NDArray: ...


def calculate_negative_flow(
    comprehensive_preference_index: npt.NDArray,
) -> npt.NDArray: ...


def calculate_net_flow(
    positive_flow: npt.NDArray, negative_flow: npt.NDArray
) -> npt.NDArray: ...


# Return type to discuss
def create_partial_ranking(net_flow: npt.NDArray) -> list[list[int]]: ...


def create_complete_ranking(
    positive_flow: npt.NDArray,
    negative_flow: npt.NDArray,
) -> list[int]: ...
