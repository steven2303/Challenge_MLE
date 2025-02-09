import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from pathlib import Path
import joblib

class DelayModel:
    def __init__(
        self
    ):
        self._model = None
        self._model_path = "models/delay_model.joblib" 
        self._categories_path = "models/categories.joblib"
        self.features_cols = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        # Constantes de validación
        self._valid_months = set(range(1, 13))
        self._valid_flight_types = {'N', 'I'}
        self._valid_airlines = None

    def _get_min_diff(self, data: pd.DataFrame) -> pd.Series:
        """Calculate time difference in minutes between Fecha-O and Fecha-I"""
        fecha_o = pd.to_datetime(data['Fecha-O'])
        fecha_i = pd.to_datetime(data['Fecha-I'])
        min_diff = ((fecha_o - fecha_i).dt.total_seconds())/60
        return min_diff

    def _save_training_categories(self, airliness: set) -> None:
        """Save valid categories from training data"""
        Path(self._categories_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'valid_airlines': airliness}, self._categories_path)

    def _load_training_categories(self) -> None:
        """Load valid categories from saved training data"""
        categories = joblib.load(self._categories_path)
        print(categories)
        self._valid_airlines = categories['valid_airlines']

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data against training categories"""
        self._load_training_categories()

        # Validar MES
        invalid_months = set(data['MES'].unique()) - self._valid_months
        if invalid_months:
            raise ValueError(f"Invalid months found: {invalid_months}")

        # Validar TIPOVUELO
        invalid_flight_types = set(data['TIPOVUELO'].unique()) - self._valid_flight_types
        if invalid_flight_types:
            raise ValueError(f"Invalid flight types found: {invalid_flight_types}")

        # Validar OPERA contra operadores del training
        invalid_airlines = set(data['OPERA'].unique()) - self._valid_airlines
        if invalid_airlines:
            raise ValueError(f"Invalid airliness found: {invalid_airlines}")

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.
        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        is_training = target_column is not None

        if is_training:
            # En training, guardamos los operadores válidos
            self._valid_airlines = set(data['OPERA'].unique())
            self._save_training_categories(self._valid_airlines)
        else:
            # En predicción, validamos contra las categorías del training
            self._validate_data(data)

        data['min_diff'] = self._get_min_diff(data)
        data['delay'] = np.where(data['min_diff'] > 15, 1, 0)

        #if is_training:
        #opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        #tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        #mes_dummies = pd.get_dummies(data['MES'], prefix='MES')

        #features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)

        # Create dummy variables
        dummy_data = pd.get_dummies(data, columns=['OPERA', 'TIPOVUELO', 'MES'])
        
        # Ensure all required columns exist with 0s if not present
        for col in self.features_cols:
            if col not in dummy_data.columns:
                dummy_data[col] = 0

        # Select only the features we need
        features = dummy_data[self.features_cols]

        if is_training:
            target = pd.DataFrame({'delay': data['delay']})
            return features, target
        
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.
        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Convert target DataFrame to array for LogisticRegression
        target_values = target.values.ravel()

        # Calculate class weights
        n_y0 = np.sum(target_values == 0)
        n_y1 = np.sum(target_values == 1)
        class_weight = {0: 1, 1: n_y0/n_y1}

        # Initialize and train the model
        self._model = LogisticRegression(
            random_state=1,
            class_weight=class_weight,
            max_iter=1000
        )
        
        self._model.fit(features, target_values)
        self.save_model(self._model_path)

    def save_model(
        self,
        filepath: str
    ) -> None:
        """
        Save the trained model to disk.
        Args:
            filepath (str): Path where the model will be saved
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(self._model, filepath)

    def load_model(
        self, 
        filepath: str
    ) -> None:
        """
        Load a trained model from disk.
        Args:
            filepath (str): Path to the saved model
        """            
        self._model = joblib.load(filepath)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.
        Args:
            features (pd.DataFrame): preprocessed data.
        Returns:
            List[int]: predicted targets.
        """
        if self._model is None:
            self.load_model(self._model_path)
        
        predictions = self._model.predict(features)
        return predictions.tolist()