import pandas as pd
import os

DATA_PATH = "data/data.csv"

def test_data_exists():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found"

def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Found {nulls} missing values"

def test_expected_columns():
    df = pd.read_csv(DATA_PATH)
    expected = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected.issubset(set(df.columns)), f"Missing expected columns: {expected - set(df.columns)}"

