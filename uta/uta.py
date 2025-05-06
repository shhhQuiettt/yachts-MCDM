import pandas as pd
# read csv

df = pd.read_csv("dataset/ahp/utility.csv")

# sort df according to Utility column
# df.sort_values(by=["Utility"], ascending=False, inplace=True)
# print the sorted dataframe
# print(df)

df.set_index(df.columns[0], inplace=True)



from promethee.utils import Relation
complete_ranking = set()

number_of_alternative = len(df.index)
print(df['Utility'].iloc[0])
for i in range(number_of_alternative):
    for j in range(i + 1, number_of_alternative):

        if df["Utility"].iloc[i] > df["Utility"].iloc[j]:
            complete_ranking.add((df.index.tolist()[i], df.index.tolist()[j], Relation.PREFERRED))
        elif df["Utility"].iloc[i] < df["Utility"].iloc[j]:
            complete_ranking.add((df.index.tolist()[j], df.index.tolist()[i], Relation.PREFERRED))
        elif df["Utility"].iloc[i] == df["Utility"].iloc[j]:
            complete_ranking.add((df.index.tolist()[i], df.index.tolist()[j], Relation.INDIFFERENT))
        else:
            complete_ranking.add((df.index.tolist()[i], df.index.tolist()[j], Relation.INCOMPARABLE))

from promethee.utils import display_ranking


display_ranking(complete_ranking, "AHP Method")

# from scipy.stats import kendalltau

# # Map item names to IDs
# names = [
#     "Laguna 25", "Maxus 24", "Antila 24", "Antila 24.4 #2", "Aquatic 25T", "Janmor 25",
#     "Antila 24.4", "Mariner 24", "Maxus 24 Evo #2", "Phobos 25", "Antila 28.8",
#     "Maxus 24 Evo", "Tes 246 Versus", "Phobos 24.5 #2", "Phobos 24.5", "Phobos 25 #2"
# ]
# name_to_index = {name: i for i, name in enumerate(names)}

# # Score-based ranking (original table)
# score_ranking = [
#     "Laguna 25", "Maxus 24", "Antila 24", "Antila 24.4 #2", "Aquatic 25T", "Janmor 25",
#     "Antila 24.4", "Mariner 24", "Maxus 24 Evo #2", "Phobos 25", "Antila 28.8",
#     "Maxus 24 Evo", "Tes 246 Versus", "Phobos 24.5 #2", "Phobos 24.5", "Phobos 25 #2"
# ]
# score_ranking_idx = [name_to_index[name] for name in score_ranking]

# # Utility-based ranking (new list)
# utility_ranking = [
#     "Laguna 25", "Antila 24.4 #2", "Maxus 24", "Antila 24", "Phobos 24.5", "Phobos 25 #2",
#     "Janmor 25", "Maxus 24 Evo #2", "Phobos 25", "Mariner 24", "Tes 246 Versus",
#     "Antila 28.8", "Phobos 24.5 #2", "Antila 24.4", "Aquatic 25T", "Maxus 24 Evo"
# ]
# utility_ranking_idx = [name_to_index[name] for name in utility_ranking]

# # Compute Kendall's tau and distance
# tau, _ = kendalltau(score_ranking_idx, utility_ranking_idx)
# n = len(score_ranking_idx)
# kendall_distance = int((1 - tau) * n * (n - 1) / 2)

# print("Kendall's tau:", round(tau, 4))
# print("Kendall's distance:", kendall_distance)
