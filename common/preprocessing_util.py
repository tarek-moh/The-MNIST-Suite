from skimage.feature import hog
import pandas as pd
import numpy as np

class PreprocessingUtil:

    """
        @params: 
            df: pd.DataFrame - The dataframe to split
            test_size: float - The size of the test set
            random_state: int - The random state to use for reproducibility
        @return: 
            train_df: pd.DataFrame - The training set
            test_df: pd.DataFrame - The test set
        
        Note:
            - split isn't randomized
    """
    @staticmethod
    def train_test_split(df: pd.DataFrame, test_size: float = 0.2):
        train_mask = df.index < (len(df) * (1 - test_size))
        test_mask = ~train_mask

        train_df = df[train_mask]
        test_df = df[test_mask]

        if 'class' in df.columns:
            x_train = train_df.drop(columns='class')
            y_train = train_df['class']
            x_test = test_df.drop(columns='class')
            y_test = test_df['class']
        else:
            x_train = train_df
            y_train = None
            x_test = test_df
            y_test = None
        
        return x_train, y_train, x_test, y_test
    
    @staticmethod
    def binarize_labels(df: pd.DataFrame, pos_digits: set):
        df_copy = df.copy()
        df_copy['class'] = df_copy['class'].apply(lambda x: 1 if x in pos_digits else -1)
        return df_copy

    @staticmethod
    def hog_feature_extractor(df: pd.DataFrame):
        print("Extracting HOG features...")
        
        if 'class' in df.columns:
            X = df.drop(columns='class').to_numpy()
            y = df['class'].to_numpy()
        else:
            X = df.to_numpy()
            y = None
        
        hog_features = []
        for i in range(X.shape[0]):
            hog_features.append(hog(X[i].reshape(28, 28), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False))
        
        hog_df = pd.DataFrame(hog_features)
        
        if y is not None:
            hog_df['class'] = y
            
        print(f"Extraction complete. New shape: {hog_df.shape}")
        return hog_df
