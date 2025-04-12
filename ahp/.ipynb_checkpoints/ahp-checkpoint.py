import pandas as pd
from pathlib import Path
from pprint import pprint
import sys
import numpy.typing as npt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def get_priorities(pairwise_comparison_matrix: npt.NDArray) -> npt.NDArray:
    for i in range(pairwise_comparison_matrix.shape[0]):
        for j in range(pairwise_comparison_matrix.shape[1]):
            assert pairwise_comparison_matrix[i, j] > 0, f"Pairwise comparison matrix must be positive, but got {pairwise_comparison_matrix[i, j]} at ({i}, {j})"
            assert abs(pairwise_comparison_matrix[i,j] - 1/pairwise_comparison_matrix[j,i]) < 1e-5, f"Pairwise comparison matrix must be reciprocal, but got {pairwise_comparison_matrix[i, j]} and {pairwise_comparison_matrix[j, i]} at ({i}, {j})"

    eigenvalues, eigenvectors = la.eig(pairwise_comparison_matrix)


    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    assert not np.iscomplex(principal_eigenvector).any()
    principal_eigenvector = principal_eigenvector.real
    
    principal_eigenvector = principal_eigenvector / np.sum(principal_eigenvector)
    assert (principal_eigenvector >= 0).all()
    return principal_eigenvector

    

def get_consistency_index(comparison_matrix: npt.NDArray):
    assert len(comparison_matrix.shape) == 2
    assert comparison_matrix.shape[0] == comparison_matrix.shape[1]


    # eigenvalue = comparison_matrix[0, :] @ priorities / priorities[0]
    n = comparison_matrix.shape[0] 
    eigenvalue = np.max(la.eigvals(comparison_matrix))
    assert not np.iscomplex(eigenvalue)
    eigenvalue = eigenvalue.real
    return (eigenvalue - n) / (n-1)


def get_randomness_index(n: int) -> float:
    # how to caclulate?
    #index 0, 1
    ris = [np.inf,np.inf, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.51]
    assert n < len(ris), f"Can only get RI of n <= {len(ris)} got: {n=}"

    return ris[n-1]


def get_relative_consistency_index(comparison_matrix: npt.NDArray) -> float:
    assert len(comparison_matrix.shape) == 2
    assert comparison_matrix.shape[0] == comparison_matrix.shape[1]

    return get_consistency_index(comparison_matrix)/ get_randomness_index(comparison_matrix.shape[0])



def recreate_consitant_matrix(priorities: npt.NDArray) -> npt.NDArray:
    return np.outer(priorities, 1/priorities)

if __name__ == "__main__":
    hierarchy = ["comparisons_category.csv", "comparisons_type.csv", "comparisons_alternatives.csv"]

    if len(sys.argv) != 2:
        print("Usage: python ahp.py <data_dir>")
        sys.exit(1)


    datadir = sys.argv[1]

    weights = {}
    for file in hierarchy:
        data = pd.read_csv(Path(datadir).joinpath(file))

        # Generating pairwise comparison matrix
        for parent in data.parent.unique():
            comparisons = pd.DataFrame(data.loc[data["parent"] == parent, :])
            print(comparisons) 
            unique_elements = list(set(comparisons["Element1"].unique().tolist() + comparisons["Element2"].unique().tolist()))

            pairwise_comparisons = pd.DataFrame(index=unique_elements, columns=unique_elements)


            for element in unique_elements:
                pairwise_comparisons.at[element, element] = 1.0
            for comparison_id in comparisons.index:
                el1 = comparisons.at[comparison_id, "Element1"]
                el2 = comparisons.at[comparison_id, "Element2"]
                judgement = comparisons.at[comparison_id, "Judgement"]
                pairwise_comparisons.at[el1, el2] = float(judgement)
                pairwise_comparisons.at[el2, el1]  = 1 / float(judgement)


            comparison_matrix_raw = pairwise_comparisons.to_numpy(dtype=float)
            print(comparison_matrix_raw)
            assert not np.isnan(comparison_matrix_raw).any()

            priorities = get_priorities(comparison_matrix_raw)

            for element_id in range(len(pairwise_comparisons.index)):
                element =  pairwise_comparisons.index[element_id]
                weights[element] = priorities[element_id].item()
             

            relative_consistency_index = get_relative_consistency_index(comparison_matrix_raw)

            # print(f"-------------------------")
            # print(f"For pairwise comparison of: {result.index} in respect to {parent}")
            # print(f"\tOriginal comparison matrix:")
            # print(comparison_matrix_raw)
            # print(f"\tRelative Consistency Index = {relative_consistency_index}")
            # print(f"\tRecreated Consistent Matrix")


    
    hierarchy=pd.read_csv(Path(datadir).joinpath("hierarchy.csv"))

    datapoints = pd.read_csv(Path(datadir).joinpath("dataset.csv"), index_col=0)
    print(weights.keys())

    #copy result but add column score
    result = datapoints.copy()
    result["score"] = np.nan


    for name, datapoint in datapoints.iterrows():
        score_comprehensive = 0.0

        for parent in datapoint.index:
            score_feature = 1.0
            
            element = datapoint[parent]
            score_feature *= weights[element]

            element = parent
            i = 0
            while True:
                print(element, type(element))
                score_feature *= weights[element]

                if i == len(hierarchy.columns) - 1:
                    break

                curr_col_name = hierarchy.columns[i]
                next_col_name = hierarchy.columns[i+1] if i < len(hierarchy.columns) else None

                element = hierarchy.loc[hierarchy[curr_col_name] == element, next_col_name].values[0]
                i+=1

            score_comprehensive += score_feature

        result.at[name, "score"] = score_comprehensive

    print(result.sort_values(by="score", ascending=False))
