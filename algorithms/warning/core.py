# 4. 主控制系统
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

from algorithms.warning.models import BroadLearningSystem
from algorithms.warning.preprocessing import FloodDataPreprocessor
from algorithms.warning.scenario import ScenarioClassifier

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
class FloodEarlyWarningSystem:
    def __init__(self, scenario_models: Dict[str, BroadLearningSystem]):
        self.preprocessor = FloodDataPreprocessor()
        self.scenario_classifier = ScenarioClassifier()
        self.scenario_models = scenario_models
        self.current_scenario = None
        self.current_model = None
        self.historical_distributions = {}

    def process_real_time_data(self, real_time_data: Dict) -> Dict:
        processed_data = self.preprocessor.temporal_alignment(real_time_data)
        weather_features = self._extract_weather_features(processed_data)
        hydro_features = self._extract_hydrology_features(processed_data)
        reservoir_features = self._extract_reservoir_features(processed_data)
        scenario, confidence = self.scenario_classifier.predict_scenario(weather_features, hydro_features, reservoir_features)

        if scenario in self.scenario_models and scenario != 'no_scenario':
            self.current_scenario = scenario
            self.current_model = self.scenario_models[scenario]
            X_new = self._prepare_model_input(processed_data, scenario)
            if X_new.size > 0:
                deviation = self._compute_distribution_deviation(X_new, scenario)
                y_new = np.random.randn(len(X_new), 4)  # 模拟标签
                if deviation < 0.2:
                    self.current_model.incremental_learning(X_new, y_new, mode='weight_update')
                else:
                    self.current_model.incremental_learning(X_new, y_new, mode='structure_expansion', n_new_nodes=5)
                if self.current_model.n_enhancement_nodes > 80:
                    self._knowledge_distillation()

        warning_levels = self._predict_warning_levels(processed_data)
        return {
            'scenario': scenario,
            'confidence': confidence,
            'warning_levels': warning_levels,
            'timestamp': datetime.now(),
            'deviation': locals().get('deviation', 0.0),
            'input_features': {
                'rainfall_mean': weather_features.get('rainfall_intensity', 0),
                'water_level_mean': hydro_features.get('water_level', 0)
            }
        }

    def _extract_weather_features(self, data):
        features = {}
        if 'weather_data' in data:
            df = data['weather_data']
            if not df.empty:
                if 'rainfall' in df.columns:
                    features['rainfall_intensity'] = df['rainfall'].max()
                    features['accumulated_rainfall_24h'] = df['rainfall'].sum()
                if 'wind_speed' in df.columns:
                    features['wind_speed'] = df['wind_speed'].mean()
        return features

    def _extract_hydrology_features(self, data):
        features = {}
        if 'hydrology_data' in data:
            df = data['hydrology_data']
            if not df.empty:
                if 'level' in df.columns:
                    features['water_level'] = df['level'].max()
                if 'flow' in df.columns:
                    features['flow_rate'] = df['flow'].max()
                if 'velocity' in df.columns:
                    features['water_velocity'] = df['velocity'].mean()
        return features

    def _extract_reservoir_features(self, data):
        features = {}
        if 'reservoir_data' in data:
            df = data['reservoir_data']
            if not df.empty:
                if 'level' in df.columns:
                    features['water_level'] = df['level'].iloc[-1]
                    features['flood_limit_level'] = 100
                if 'outflow' in df.columns:
                    features['outflow'] = df['outflow'].iloc[-1]
                if 'inflow' in df.columns:
                    features['inflow'] = df['inflow'].iloc[-1]
                features['has_schedule'] = False
        return features

    def _prepare_model_input(self, data, scenario):
        feature_mapping = {
            'flash_flood': ['rainfall','terrain','soil'],
            'typhoon_flood': ['wind','rainfall','vegetation'],
            'reservoir_flood': ['outflow','inflow','slope'],
            'urban_flood': ['rainfall','drainage','impervious'],
            'landslide_flood': ['rainfall','slope','soil'],
            'channel_flood': ['level','flow','capacity'],
            'dam_break': ['level','integrity','pressure'],
            'climate_change_flood': ['temperature','sea_level','extreme']
        }
        features = []
        if scenario in feature_mapping:
            for feat in feature_mapping[scenario]:
                found = False
                for data_key, df in data.items():
                    if not df.empty and feat in data_key.lower():
                        values = df.values
                        if values.size > 0:
                            features.append(values.flatten()[:10].mean())
                            found = True
                            break
                if not found:
                    features.append(0.0)
        if len(features) < 3:
            features.extend([0.0] * (3 - len(features)))
        return np.array(features).reshape(1,-1)

    def _compute_distribution_deviation(self, X_new, scenario):
        if X_new.size == 0:
            return 0.0
        if scenario not in self.historical_distributions:
            self.historical_distributions[scenario] = np.random.randn(100, X_new.shape[1])
        hist_mean = np.mean(self.historical_distributions[scenario], axis=0)
        new_mean = np.mean(X_new, axis=0)
        return np.sqrt(np.mean((new_mean - hist_mean)**2))

    def _predict_warning_levels(self, data):
        if self.current_model is None or self.current_scenario is None:
            return {}
        X_input = self._prepare_model_input(data, self.current_scenario)
        if X_input.size == 0:
            return {}
        predictions = self.current_model.predict(X_input)
        if predictions.size == 0:
            return {}
        pred_value = np.mean(predictions)
        if pred_value < 0.25:
            level = '蓝'
        elif pred_value < 0.5:
            level = '黄'
        elif pred_value < 0.75:
            level = '橙'
        else:
            level = '红'
        warning_levels = {}
        for i in range(12):
            warning_levels[f'station_{i+1}'] = level
        return warning_levels

    def _knowledge_distillation(self):
        if self.current_model and self.current_model.n_enhancement_nodes > 80:
            n_keep = int(self.current_model.n_enhancement_nodes * 0.6)
            self.current_model.n_enhancement_nodes = n_keep

def build_warning_system():
    scenario_models = {
        'flash_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'typhoon_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'reservoir_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'urban_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'landslide_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'channel_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'dam_break': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30),
        'climate_change_flood': BroadLearningSystem(n_feature_nodes=10, n_enhancement_nodes=30)
    }

    for model in scenario_models.values():
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100, 4)
        model.fit(X_train, y_train)

    return FloodEarlyWarningSystem(scenario_models)