# """
# This script computes and saves simple features 
# """


# import pandas as pd
# from src.dataloader import ChallengeDataLoader
# from src.utils import Cols

# from pathlib import Path



# PATH_DATA = Path().cwd().parent / "data"

# # def check_features:

# def _compute_turnover_adj_sv(X):
#     return X[Cols.vol_cols()].div(X[Cols.TURNOVER()].values).add_prefix("turnover_adj_")

# def main():
#     X_train = ChallengeDataLoader.load_X_train()
#     X_test = ChallengeDataLoader.load_X_test()

#     X_train = pd.concatenate((X_train, _compute_turnover_adj_sv(X_train)), axis=1)
#     X_test = pd.concatenate((X_test, _compute_turnover_adj_sv(X_test)), axis=1)

#     X_train.to_csv(PATH_DATA / "X_train_features.csv")
#     X_test.to_csv(PATH_DATA / "X_test_features.csv")
               
#     return 

# if __name__=="__main__":
#     main()
