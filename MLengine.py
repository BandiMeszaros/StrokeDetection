import json
import pandas as pd
import pickle
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import utils


class MLEngine(object):
    def __init__(self):
        self.encoders = self.load("artifacts/encoders.pkl")
        self.imputer = self.load("artifacts/imputer.pkl")
        self.scalers = self.load("artifacts/scalers.pkl")
        self.model_GB = self.load("models/best_model_GradientBoosting.pkl")
        self.model_KNN = self.load("models/best_model_KNN.pkl")
        self.model_RF = self.load("models/best_model_RandomForest.pkl")
        self.model_SVM = self.load("models/best_model_SVM.pkl")

        self.models = {"GradientBoosting": self.model_GB,
                       "KNN": self.model_KNN,
                       "RandomForest": self.model_RF,
                       "SVM": self.model_SVM}

    def predict(self, data_row: pd.DataFrame):
        """
        Predicts the target variable for a given data row.

        Parameters:
            data_row (pd.DataFrame): A list containing the input data for prediction.

        Returns:
            tuple: A tuple containing the predicted target variable as a JSON string and the HTTP status code.
        """
        try:
            df = self.preprocess_inference(data_row)
            target_column, input_array = self.get_target_column(df)
            if target_column is not None:
                y_pred = {}
                for name, model in self.models.items():
                    y_pred[name] = int(model.predict(input_array))
            else:
                raise Exception("Target column not found in dataframe.")
            return_json = {"prediction": y_pred, "target_column": target_column[0]}
            return json.dumps(return_json), 200

        except Exception as e:
            return json.dumps({'message': 'Internal Server Error. ', 'error': str(e)}), 500

    def preprocess_train(self, df):

        """
        Preprocesses the input data for training, handling missing values
        and transforming the data using encoders and scalers.

        Parameters:
            df(pd.DataFrame): input data to be preprocessed

        Returns:
            data(pd.DataFrame): preprocessed data ready for training
        """
        self.imputer = SimpleImputer(strategy='mean')
        if df.isnull().values.any():
            cols_name = df.columns[df.isnull().any()].tolist()
            for col in cols_name:
                df[col] = self.imputer.fit_transform(df[[col]])

        df = utils.create_new_features(df)
        self.encoders = {}
        for col in utils.NOMINAL_COLS:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            new_data = encoder.fit_transform(df[col].to_numpy().reshape(-1, 1))

            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            df = pd.concat([df, new_df], axis=1)
            self.encoders[col] = encoder

        df.drop(columns=utils.NOMINAL_COLS, inplace=True)

        for col in utils.CONTINUOUS_COLS:
            df[col + '_zscore'] = stats.zscore(df[col])
            outliers_indices = df[abs(df[col + '_zscore']) > 3].index
            mean = df[col].mean()
            df.loc[outliers_indices, col] = mean
            df.drop(columns=[col + '_zscore'], inplace=True)

        self.scalers = {}
        for col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler

       # save encoders, imputer and scalers
        self._save(self.encoders, "artifacts/encoders.pkl")
        self._save(self.imputer, "artifacts/imputer.pkl")
        self._save(self.scalers, "artifacts/scalers.pkl")

        X = df.drop('stroke', axis=1)
        y = df['stroke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # save train and test data
        self._save_csv(X_train, "engine_data/X_train.csv")
        self._save_csv(X_test, "engine_data/X_test.csv")
        self._save_csv(y_train, "engine_data/y_train.csv")
        self._save_csv(y_test, "engine_data/y_test.csv")

        return X_train, X_test, y_train, y_test
    @staticmethod
    def _save(data, file_path):
        """
        A static method to save data to a file using pickle.

        Parameters:
            data: The data to save.
            file_path (str): The path to the file to save the data to.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def _save_csv(data, file_path):
        """
        A static method to save data to a csv file.

        Parameters:
            data: The data to save.
            file_path (str): The path to the file to save the data to.
        """
        data.to_csv(file_path, index=False)

    def preprocess_inference(self, data: pd.DataFrame):
        """
        Preprocesses the input data for inference, handling missing values
        and transforming the data using encoders and scalers.

        Parameters:
            data(pd.DataFrame): input data to be preprocessed

        Returns:
            data(pd.DataFrame): preprocessed data ready for inference
        """
        if data.isnull().values.any():
            cols_name = data.columns[data.isnull().any()].tolist()
            for col in cols_name:
                data[col] = self.imputer.transform(data[col])

        data = utils.create_new_features(data)

        for col, encoder in self.encoders.items():
            new_data = encoder.transform(data[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            data = pd.concat([data, new_df], axis=1).drop(columns=[col])

        for col, scaler in self.scalers.items():
            if col in data.columns:
                data[col] = scaler.transform(data[[col]])[0][0]
        return data

    @staticmethod
    def get_target_column(df):
        """
        A function that checks if "stroke" is in the columns of the provided dataframe and returns
        the "stroke" column along with the dataframe with "stroke" column dropped if it exists.

        Parameters:
        - self: The instance of the class.
        - df: The pandas DataFrame to operate on.

        Returns:
        - Tuple: A tuple containing the "stroke" column (if present) and the dataframe with "stroke" column dropped.
        """

        if "stroke" in df.columns:
            input_array = df.drop(columns=["stroke"])
            return df["stroke"], input_array
        else:
            return None, df


    @staticmethod
    def load(file_path):
        """
        A static method to load data from a file using pickle.

        Parameters:
            file_path (str): The path to the file to load.

        Returns:
            The data loaded from the file.
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError as err:
            print(f"File not found, error message: {err}")

        return data

    @staticmethod
    def _optimizing_model_parameters(model_name, model, param_distributions,
                                    X_train, y_train, X_validation,
                                    y_validation) -> list:
        rand_search = RandomizedSearchCV(model,
                                         param_distributions, cv=5, scoring='accuracy',
                                         n_jobs=-1, n_iter=50)
        rand_search.fit(X_train, y_train.values.ravel())
        best_model = rand_search.best_estimator_
        y_pred = best_model.predict(X_validation)
        accuracy = accuracy_score(y_validation, y_pred)
        train_accuracy = accuracy_score(y_train, best_model.predict(X_train))

        # Enhanced output
        print(f"\nOptimized Parameters for {model_name}:")
        print(rand_search.best_params_)
        print(f"Optimized Accuracy (train) ({model_name}): {train_accuracy}")
        print(f"Optimized Accuracy (validation) ({model_name}): {accuracy}")

        # Unchanged saving logic
        with open(f"models/{model_name}.json", "w") as f:
            json.dump([rand_search.best_params_, {"accuracy_validation": accuracy},
                       {"accuracy_train": train_accuracy}],
                      f)

        return [rand_search.best_params_, {"accuracy_validation": accuracy},
                       {"accuracy_train": train_accuracy}]

    def train(self, data: pd.DataFrame):
        """
        Trains the model with the provided data.

        Parameters:
            data(pd.DataFrame): input data to be trained on
        """
        model_fits = {}
        try:
            X_train, X_test, y_train, y_test = self.preprocess_train(data)
            for name, (model, grid) in utils.USED_MODELS.items():
                best_fit = self._optimizing_model_parameters(name, model(),
                                                             grid, X_train,
                                                             y_train, X_test,
                                                             y_test)
                model_fits[name] = best_fit
        except Exception as e:
            return json.dumps({'message': 'Internal Server Error. ', 'error': str(e)}), 500
        return json.dumps(model_fits), 200


engine = MLEngine()
