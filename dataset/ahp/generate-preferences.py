import pandas as pd
import numpy as np
import random
from itertools import combinations

def generate_preferences(data, parent: str) -> pd.DataFrame:
    col = data[parent].unique().tolist()
    minim = np.min(col)
    maxim = np.max(col)
    new_col = {}
    for el1, el2 in combinations(col, 2):
        el1,el2 = (int(el1), int(el2)) if int(el1) > int(el2) else (int(el2), int(el1))

        comp = (el1-el2) / (maxim-minim) + random.uniform(-0.2, 0.2)
        preference = 1 + 8 * comp
        preference = max(1, min(int(preference), 9))

        new_col[(el1, el2)] = preference

    print(new_col)
    # create list of tuples
    preferences = [ (parent, e1, e2, p) for (e1, e2), p in new_col.items() ]

    return pd.DataFrame(preferences, columns=["parent", "Element1", "Element2", "Judgement"])



if __name__ == "__main__":
    data = pd.read_csv("./dataset.csv", index_col=0)

    preferences = pd.DataFrame(columns=["parent", "Element1", "Element2", "Judgment"])

    ppd = generate_preferences(data, "PricePerDay")

    dp = generate_preferences(data, "Deposit")

    ep = generate_preferences(data, "EnginePower")

    sa = generate_preferences(data, "SailArea")

    wi = generate_preferences(data, "Width")


    preferences = pd.concat([preferences, ppd, dp, ep, sa, wi], ignore_index=True)

    # save
    preferences.to_csv("comparisons_alternatives.csv", index=False)
    print(preferences)




