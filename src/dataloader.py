from pathlib import Path
import pandas as pd 

PATH_DATA = Path(__file__).resolve().parent.parent / "data"


class ChallengeDataLoader:

    PATH_X_TRAIN = PATH_DATA / "X_train.csv"
    PATH_Y_TRAIN = PATH_DATA / "y_train.csv"
    PATH_X_TEST = PATH_DATA / "X_test.csv"
    PATH_SAMPLE_SUBMISSION = PATH_DATA / "sample_submission.csv" 
    
    @classmethod
    def load_X_train(cls):
        return pd.read_csv(cls.PATH_X_TRAIN,index_col='ROW_ID')
    
    @classmethod
    def load_X_test(cls):
        return pd.read_csv(cls.PATH_X_TEST,index_col='ROW_ID')
    
    @classmethod
    def load_y_train(cls):
        y_train = pd.read_csv(cls.PATH_Y_TRAIN,index_col='ROW_ID')
        return y_train.assign(target_binarized=(y_train > 0).astype(int))
    
    @classmethod
    def load_train_df(cls):
        X_train = cls.load_X_train()
        y_train = cls.load_y_train()
        return pd.concat((X_train, y_train), axis=1)
    
    @classmethod
    def load_sample_submission(cls):
        return pd.read_csv(cls.PATH_SAMPLE_SUBMISSION,index_col='ROW_ID')
         