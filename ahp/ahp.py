import pandas as pd
from pathlib import Path
import sys
import numpy.typing as npt
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def get_priorites(pairwise_comparison_matrix: npt.NDArray) -> npt.NDArray:
    for i in range(pairwise_comparison_matrix.shape[0]):
        for j in range(pairwise_comparison_matrix.shape[1]):
            assert pairwise_comparison_matrix[i, j] > 0, f"Pairwise comparison matrix must be positive, but got {pairwise_comparison_matrix[i, j]} at ({i}, {j})"
            assert abs(pairwise_comparison_matrix[i,j] - 1/pairwise_comparison_matrix[j,i]) < 1e-5, f"Pairwise comparison matrix must be reciprocal, but got {pairwise_comparison_matrix[i, j]} and {pairwise_comparison_matrix[j, i]} at ({i}, {j})"

    eigenvalues, eigenvectors = la.eig(pairwise_comparison_matrix)


    principal_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    assert not np.iscomplex(principal_eigenvector).any()
    principal_eigenvector = principal_eigenvector.astype(float)
    
    principal_eigenvector = principal_eigenvector / np.sum(principal_eigenvector)
    return principal_eigenvector

    



if __name__ == "__main__":
    hierarchy = ["comparisons_type.csv", "comparisons_alternatives.csv"]

    if len(sys.argv) != 2:
        print("Usage: python ahp.py <data_dir>")
        sys.exit(1)


    datadir = sys.argv[1]

    for file in hierarchy:
        data = pd.read_csv(Path(datadir).joinpath(file))

        # Generating pairwise comparison matrix
        for parent in data.parent.unique()[::-1]:
            comparisons = pd.DataFrame(data.loc[data["parent"] == parent, :])
            
            unique_elements = list(set(comparisons["Element1"].unique().tolist() + comparisons["Element2"].unique().tolist()))

            result = pd.DataFrame(index=unique_elements, columns=unique_elements)

            for element in unique_elements:
                result.at[element, element] = 1.0
            for comparison_id in comparisons.index:
                el1 = comparisons.at[comparison_id, "Element1"]
                el2 = comparisons.at[comparison_id, "Element2"]
                judgement = comparisons.at[comparison_id, "Judgement"]
                result.at[el1, el2] = float(judgement)
                result.at[el2, el1]  = 1 / float(judgement)


            comparison_matrix_raw = result.to_numpy(dtype=float)
            assert not np.isnan(comparison_matrix_raw).any()

            priorities = get_priorites(comparison_matrix_raw)
            print(priorities)
            sys.exit(0)



         







