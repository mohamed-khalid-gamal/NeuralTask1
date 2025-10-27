
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    def __init__(self, filepath='penguins.csv'):
        self.filepath = filepath
        self.data = None
        self.label_encoder = LabelEncoder()
        self.feature_names = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        self.class_names = ['Adelie', 'Chinstrap', 'Gentoo']
        self.feature_means = {}
        self.feature_stds = {}

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    def preprocess(self):
        if self.data is None:
            self.load_data()

        for col in ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col].fillna(self.data[col].mean(), inplace=True)

        self.data['OriginLocation'] = self.label_encoder.fit_transform(self.data['OriginLocation'])

        return self.data

    def normalize_features(self, X, feature1, feature2):

        X_normalized = X.copy()

        for i, feat in enumerate([feature1, feature2]):
            mean = X[:, i].mean()
            std = X[:, i].std()
            if std > 0:
                X_normalized[:, i] = (X[:, i] - mean) / std
            self.feature_means[feat] = mean
            self.feature_stds[feat] = std

        return X_normalized

    def get_class_data(self, class1, class2, feature1, feature2, normalize=False):

        if self.data is None:
            self.preprocess()

        class_data = self.data[self.data['Species'].isin([class1, class2])].copy()

        X = class_data[[feature1, feature2]].values

        if normalize:
            X = self.normalize_features(X, feature1, feature2)

        y = class_data['Species'].apply(lambda x: -1 if x == class1 else 1).values

        return X, y

    def split_data(self, X, y, train_size=30, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        class1_indices = np.where(y == -1)[0]
        class2_indices = np.where(y == 1)[0]

        np.random.shuffle(class1_indices)
        np.random.shuffle(class2_indices)

        train_idx_c1 = class1_indices[:train_size]
        test_idx_c1 = class1_indices[train_size:]

        train_idx_c2 = class2_indices[:train_size]
        test_idx_c2 = class2_indices[train_size:]

        train_indices = np.concatenate([train_idx_c1, train_idx_c2])
        test_indices = np.concatenate([test_idx_c1, test_idx_c2])

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test
