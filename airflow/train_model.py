import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow
from mlflow.models import infer_signature
 
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "insurance_clean.csv")
    model_path = os.path.join(base_path, "gb_insurance.pkl")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")

    print("Загрузка данных...")
    df = pd.read_csv(data_path)

    X = df.drop("charges", axis=1)
    y = df["charges"]

    cat_features = ['sex', 'smoker', 'region']
    num_features = ['age', 'bmi', 'children']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5],
    }

    mlflow.set_experiment("Insurance_Model_Training")

    with mlflow.start_run():
        print("Обучение модели...")
        clf = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        clf.fit(X_train, y_train)

        best_model = clf.best_estimator_
        y_pred = best_model.predict(X_val)

        rmse, mae, r2 = eval_metrics(y_val, y_pred)
        print(f"Оценка модели: RMSE={rmse}, MAE={mae}, R2={r2}")

        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({'rmse': rmse, 'mae': mae, 'r2': r2})

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        joblib.dump(best_model, model_path)
        print(f"Модель сохранена в: {model_path}")
