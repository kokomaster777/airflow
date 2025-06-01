import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train_model
import os
 
def download_data():
    base_path = os.path.dirname(__file__)
    input_path = os.path.join(base_path, 'insurance_miptstats.csv')
    output_path = os.path.join(base_path, 'insurance_raw.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print("Downloaded data saved to:", output_path)
    print(df.head())

def clear_data():
    base_path = os.path.dirname(__file__)
    input_path = os.path.join(base_path, 'insurance_raw.csv')
    output_path = os.path.join(base_path, 'insurance_clean.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path)
    
    # Возраст
    df['age'] = datetime.today().year - pd.to_datetime(df['birthday']).dt.year
    df.drop(columns=['birthday'], inplace=True)
    
    # Удаление выбросов
    df = df[(df['age'] >= 18) & (df['age'] <= 80)]
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    df = df[(df['charges'] >= 1000) & (df['charges'] <= 50000)]
    
    # Кодирование категориальных признаков
    cat_columns = ['sex', 'smoker', 'region']
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    df.to_csv(output_path, index=False)
    print("Cleaned data saved to:", output_path)
    print("Cleaned data shape:", df.shape)
    return True

dag_insurance = DAG(
    dag_id="insurance_training_pipe",
    start_date=datetime(2025, 4, 24),
    schedule_interval=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)

download_task = PythonOperator(
    task_id="download_insurance_data",
    python_callable=download_data,
    dag=dag_insurance
)

clear_task = PythonOperator(
    task_id="clean_insurance_data",
    python_callable=clear_data,
    dag=dag_insurance
)

train_task = PythonOperator(
    task_id="train_insurance_model",
    python_callable=train_model,
    dag=dag_insurance
)

download_task >> clear_task >> train_task
