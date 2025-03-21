import pandas as pd

def refactor_home(home):
    if home == "@":
        return 1
    else: return 0


def refactor_result(result):
    if result.split(" ")[0] == "W":
        return 1
    else: return 0

file = pd.read_csv("datasets/doncic_cut.csv")

file["WIN"] = file["R"].apply(refactor_result)
file = file.drop(columns=["R"])

file["HOME"] = file["HOME"].apply(refactor_home)

file.to_csv("./datasets/doncic_cut_ref.csv", index=False)

file = file.drop(
    columns=["MP", "ORB", "DRB", "Date", "G", "TOV", "Rk", "Tm", "Age", "GS", "FG", "FGA", "FT%", "FG%", "3PA", "FT",
             "FTA", "STL", "BLK", "OPP", "PF", "+/-", "HOME"]
)

testLength: int = 50

file.to_csv("./datasets/doncic_ref.csv", index=False)

test = file.tail(testLength)
test.to_csv("./datasets/doncic_ref_test.csv", index=False)

train = file.head(len(file) - testLength)
train.to_csv("./datasets/doncic_ref_train.csv", index=False)
