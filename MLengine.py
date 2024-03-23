import json
import os
import pandas as pd
import pickle

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

    def predict(self, data_row):
        """
        Predicts the target variable for a given data row.

        Parameters:
            data_row (list): A list containing the input data for prediction.

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
            return json.dumps(y_pred), 200

        except Exception as e:
            return json.dumps({'message': 'Internal Server Error. ', 'error': str(e)}), 500

    def preprocess_train(self, df):
        pass

    def preprocess_inference(self, data):
        """
        Preprocesses the input data for inference, handling missing values
        and transforming the data using encoders and scalers.

        Parameters:
            data: input data to be preprocessed

        Returns:
            sample_df: preprocessed data ready for inference
        """
        sample_df = pd.DataFrame([data])
        if sample_df.isnull().values.any():
            cols_name = sample_df.columns[sample_df.isnull().any()].tolist()
            for col in cols_name:
                sample_df[col] = self.imputer.transform(sample_df[col])

        sample_df = utils.create_new_features(sample_df)

        for col, encoder in self.encoders.items():
            new_data = encoder.transform(sample_df[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=encoder.get_feature_names_out([col]))
            sample_df = pd.concat([sample_df, new_df], axis=1).drop(columns=[col])

        for col, scaler in self.scalers.items():
            new_data = scaler.transform(sample_df[col].to_numpy().reshape(-1, 1))
            new_df = pd.DataFrame(new_data, columns=scaler.get_feature_names_out([col]))
            sample_df = pd.concat([sample_df, new_df], axis=1).drop(columns=[col])

        return sample_df

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
            return df["stroke"], df.drop(columns="stroke", inplace=True, axis=1)
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



