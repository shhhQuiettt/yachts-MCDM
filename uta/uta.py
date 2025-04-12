import pandas as pd
# read csv

df = pd.read_csv("dataset/uta/utility.csv")

# print(df.head())

# set frist column as index
df.set_index(df.columns[0], inplace=True)

# print(df.head())

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
# from promethee.main import create_complete_ranking
# create complete ranking
# ranking = create_complete_ranking(df)
# print(ranking)

# print(ranking.iloc[0] > ranking.iloc[1])

display_ranking(complete_ranking, "UTA Method")



