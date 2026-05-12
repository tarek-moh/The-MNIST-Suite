from skimage.feature import hog
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader


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

        """
        Takes extracted NumPy features and labels, shuffles them, and splits them 
        into Train, Validate, and Test sets based on the provided percentages.
        """

    @staticmethod
    def train_validate_test_split(X: np.ndarray, Y: np.ndarray, train_size=0.70, val_size=0.15, test_size=0.15,
                                  random_seed=42):
        # throw an error if the passed values dont add up to 100%
        assert np.isclose(train_size + val_size + test_size, 1.0), "Split percentages must sum to 1.0!"
        # Generate a list of shuffled indices
        np.random.seed(random_seed)  # Set a seed so the random split is reproducible
        shuffled_indices = np.random.permutation(len(X))

        # Apply the shuffled indices to both X and Y simultaneously to make sure that the x values match their original y values
        X_shuffled = X[shuffled_indices]
        Y_shuffled = Y[shuffled_indices]

        # Calculate the exact row numbers where we need to cut the data
        train_end = int(
            len(X) * train_size)  # use int () to make sure teh index is an integer number not a float or anything else
        val_end = train_end + int(len(X) * val_size)

        # Slice the arrays into their final sets
        X_train = X_shuffled[:train_end]
        Y_train = Y_shuffled[:train_end]

        X_val = X_shuffled[train_end:val_end]
        Y_val = Y_shuffled[train_end:val_end]

        X_test = X_shuffled[val_end:]
        Y_test = Y_shuffled[val_end:]
        # return the split data
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    """
        Downloads MNIST dataset and  processes it through a pre-trained ResNet-18 CNN model, 
        and returns the entire dataset as 512-dimensional extracted features.
    """

    @staticmethod
    def extract_features_resnet_cnn(root_dir='./data', batch_size=64):
        print("data transformation is starting...")
        resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # download MNIST
        train_set = datasets.MNIST(root=root_dir, train=True, download=True, transform=resnet_transform)
        test_set = datasets.MNIST(root=root_dir, train=False, download=True, transform=resnet_transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        # prepare resnet18 model
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # use imagenet pretrained weights
        model.fc = nn.Identity()  # Strip final layer
        model.eval()  # Set to evaluation mode

        # helper function for extracting features
        def _extract_features(dataloader, dataset_name):
            X_features, Y_labels = [], []
            print(f"Starting {dataset_name} extraction... this will take a few minutes!")

            with torch.no_grad():
                for images, labels in dataloader:
                    features = model(images)
                    X_features.append(features.numpy())  # convert from tensor to numpy
                    Y_labels.append(labels.numpy())

            return np.vstack(X_features), np.concatenate(Y_labels)

        # extract features
        X_train, y_train = _extract_features(train_loader, 'train_set')
        X_test, y_test = _extract_features(test_loader, 'test_set')

        print("Combining extracted features into a single dataset...")  # to use train validate test later
        X_all = np.vstack((X_train, X_test))
        Y_all = np.concatenate((y_train, y_test))

        print(f"Extraction Complete! Final X shape: {X_all.shape}")

        return X_all, Y_all

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
            hog_features.append(
                hog(X[i].reshape(28, 28), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
                    block_norm='L2-Hys', visualize=False))

        hog_df = pd.DataFrame(hog_features)

        if y is not None:
            hog_df['class'] = y

        print(f"Extraction complete. New shape: {hog_df.shape}")
        return hog_df