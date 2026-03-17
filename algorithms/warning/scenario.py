import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

import matplotlib
# 3. 情景判别模块
class RulesEngine:
    def filter_scenarios(self, weather_data: Dict, hydrology_data: Dict, reservoir_data: Dict) -> List[str]:
        all_scenarios = ['flash_flood','landslide_flood','urban_flood','channel_flood',
                         'reservoir_flood','dam_break','typhoon_flood','climate_change_flood']
        candidate_scenarios = all_scenarios.copy()
        accumulated_rainfall = weather_data.get('accumulated_rainfall_24h', 0)
        if accumulated_rainfall < 10:
            to_remove = ['flash_flood','typhoon_flood','urban_flood','landslide_flood']
            candidate_scenarios = [s for s in candidate_scenarios if s not in to_remove]
        reservoir_level = reservoir_data.get('water_level', 0)
        flood_limit = reservoir_data.get('flood_limit_level', 100)
        has_schedule = reservoir_data.get('has_schedule', False)
        if reservoir_level < flood_limit * 0.8 and not has_schedule:
            if 'reservoir_flood' in candidate_scenarios:
                candidate_scenarios.remove('reservoir_flood')
        current_month = datetime.now().month
        if current_month not in [6,7,8,9]:
            if 'typhoon_flood' in candidate_scenarios:
                candidate_scenarios.remove('typhoon_flood')
        if 'dam_level' not in reservoir_data:
            if 'dam_break' in candidate_scenarios:
                candidate_scenarios.remove('dam_break')
        return candidate_scenarios

class ScenarioClassifier:
    def __init__(self):
        self.rules_engine = RulesEngine()
        self.rf_classifier = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42)
        self.scenario_features = {
            'flash_flood': ['rainfall_intensity','terrain_relief','soil_moisture'],
            'typhoon_flood': ['wind_speed','accumulated_rainfall','vegetation_index'],
            'reservoir_flood': ['outflow','inflow','channel_slope'],
            'urban_flood': ['rainfall_intensity','drainage_capacity','impervious_rate'],
            'landslide_flood': ['rainfall_intensity','slope_angle','soil_saturation'],
            'channel_flood': ['water_level','flow_rate','channel_capacity'],
            'dam_break': ['dam_level','structural_integrity','water_pressure'],
            'climate_change_flood': ['temperature_anomaly','sea_level_rise','extreme_event_freq']
        }
        self._train_classifier()

    def _train_classifier(self):
        n_samples = 1000
        n_features = 12
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 8, n_samples)
        self.rf_classifier.fit(X_train, y_train)
        self.scenario_names = list(self.scenario_features.keys())

    def predict_scenario(self, weather_data: Dict, hydrology_data: Dict, reservoir_data: Dict) -> Tuple[str, float]:
        candidate_scenarios = self.rules_engine.filter_scenarios(weather_data, hydrology_data, reservoir_data)
        if not candidate_scenarios:
            return 'no_scenario', 0.0
        if len(candidate_scenarios) == 1:
            return candidate_scenarios[0], 1.0
        feature_vector = self._extract_features(weather_data, hydrology_data, reservoir_data, candidate_scenarios)
        if len(feature_vector) != 12:
            if len(feature_vector) < 12:
                feature_vector = np.pad(feature_vector, (0, 12 - len(feature_vector)), 'constant')
            else:
                feature_vector = feature_vector[:12]
        probabilities = np.zeros(8)
        all_probs = self.rf_classifier.predict_proba([feature_vector])[0]
        for i, scenario in enumerate(self.scenario_names):
            if scenario in candidate_scenarios:
                probabilities[i] = all_probs[i]
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        scenario_idx = np.argmax(probabilities)
        return self.scenario_names[scenario_idx], probabilities[scenario_idx]

    def _extract_features(self, weather_data, hydrology_data, reservoir_data, scenarios) -> np.ndarray:
        base_features = [
            weather_data.get('rainfall_intensity', 0),
            weather_data.get('accumulated_rainfall_24h', 0),
            weather_data.get('wind_speed', 0),
            hydrology_data.get('water_level', 0),
            hydrology_data.get('flow_rate', 0),
            hydrology_data.get('water_velocity', 0),
            reservoir_data.get('water_level', 0),
            reservoir_data.get('outflow', 0),
            reservoir_data.get('inflow', 0),
            0,0,0
        ]
        return np.array(base_features)