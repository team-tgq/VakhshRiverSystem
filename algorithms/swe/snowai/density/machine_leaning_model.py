
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import set_config
from ..utils import clean_cache
from ..utils.sturm_model_utils import validate_snow_class
from ..utils.xgboost_utils import download_model, validate_DOY
# 当前文件路径
current_file = Path(__file__)

# 父目录
parent = current_file.parent

VALID_SNOW_CLASSES = ['alpine', 'maritime', 'prairie', 'tundra', 'taiga', 'ephemeral']

set_config(transform_output='pandas')
# 给定一条（或多条）样本的气象/地形/雪层信息，预测雪密度（代码注释写的是 g/cm³）。

class MachineLearningDensity:
    def __init__(self, model_location: str = None, return_type: str = 'numpy'):
        self.model_location = model_location
        self.return_type = return_type
        self.model = self.load_model()

    def load_model(self):

        if self.model_location is None:
            model_path = parent / "density_model.ubj"
            self.model_location = model_path
            # self.model_location = download_model()

        xgb_model = xgb.Booster()
        xgb_model.load_model(self.model_location)  # Load models from Uiversal Binary JSON file
        
        return xgb_model
    
    def preprocess_data(self, input_data):

        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the pickle file
        pickle_path = os.path.join(current_dir, '..', 'utils', 'preprocessor', 'feature_engineering_pipeline.joblib')

        preprocessor=joblib.load(pickle_path)
        return preprocessor.transform(input_data)

    def predict(
        self,
        data: pd.DataFrame,
        snow_class: str,
        elevation: str,
        snow_depth: str,
        tavg: str,
        tmin: str,
        tmax: str,
        DOY: str
     ) -> np.ndarray | pd.Series:
        """
        A function to compute snow density using the machine learning models.

        Parameters:
        ===========
            * data (pd.DataFrame): Input dataset containing the required columns.
            * snow_class (str): Column name for snow class.
            * elevation (str): Column name for elevation in meters.
            * snow_depth (str): Column name for snow depth in meters.
            * tavg (str): Column name for average temperature in Celsius.
            * tmin (str): Column name for minimum temperature in Celsius.
            * tmax (str): Column name for maximum temperature in Celsius.
            * DOY (str): Column name for the day of the year (defaults to October 1 as origin).

            snow_class：雪类（alpine/maritime/...）
            elevation：海拔（m）
            snow_depth：雪深（m）
            tavg：平均温度（°C）
            tmin：最低温（°C）
            tmax：最高温（°C）
            DOY：日期/日序（该库里会按“10月1日为起点”转换）

        Returns:
        ========
            * np.ndarray | pd.Series: The function returns the snow density in g/cm^3.

        """

        # validate retune type
        if self.return_type.lower() not in ['numpy', 'pandas']:
            raise ValueError("Unsupported return type. Choose either 'numpy' or 'pandas'.")
        
        # Create a DataFrame from the input data
        try:
            input_data = pd.DataFrame(
                {
                    'Snow_Class': data[snow_class],
                    'Elevation': data[elevation],
                    'Snow_Depth': data[snow_depth]*100,
                    'TAVG': data[tavg],
                    'TMIN': data[tmin],
                    'TMAX': data[tmax],
                    'DOY': data[DOY]
                }
            )
        except KeyError as e:
            raise ValueError(f"Column {e.args[0]} is missing in data.")

        # Check for NaN values in the extracted columns
        if input_data.isna().any().any():
            raise ValueError("Input data contains NaN values.")
        
        # validate snow class and DOY
        # data_for_preprocessing=(
        #     input_data
        #     .assign(
        #         Snow_Class=lambda x: np.char.title(validate_snow_class(x['Snow_Class'], VALID_SNOW_CLASSES)),
        #         DOY=lambda x: validate_DOY(x['DOY'], origin=10)
        #     )
        # )

        data_for_preprocessing = (
            input_data
            .assign(
                Snow_Class=lambda x: pd.Series(
                    validate_snow_class(x['Snow_Class'], VALID_SNOW_CLASSES)
                ).str.title(),
                DOY=lambda x: validate_DOY(x['DOY'], origin=10)
            )
        )
        
        # Preprocess the input data
        preprocessed_data = self.preprocess_data(input_data=data_for_preprocessing)
        xgb_input = xgb.DMatrix(preprocessed_data)

        density = self.model.predict(xgb_input)

        if self.return_type.lower() == 'numpy':
            return density
        else:
            return pd.Series(density, index=data.index)
        
    
    def clear_cache(self):
        clean_cache('density')
