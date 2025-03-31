import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

class MLModel:
    def __init__(self, data_dir, feature_cols, target_col):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_dir, "lifestyle_data.csv"))
        X = df[self.feature_cols]
        y = df[self.target_col]
        return X, y

    def preprocess_data(self, X):
        return self.scaler.fit_transform(X)
    
    def objective(self, trial, X_train, y_train):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        }
        
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_train)[:, 1]
        score = roc_auc_score(y_train, y_pred)
        return score
    
    def train(self):
        X, y = self.load_data()
        X = self.preprocess_data(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, self.X_train, self.y_train), n_trials=50)
        self.best_params = study.best_params
        print("Best parameters: ", self.best_params)

        self.model = XGBClassifier(**self.best_params)
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X_new):
        X_new = self.scaler.transform(X_new)
        return self.model.predict_proba(X_new)[:, 1]
    
    def evaluate(self):
        y_pred = self.model.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_pred)
        print(f"ROC AUC Score: {auc_score}")
        return auc_score
