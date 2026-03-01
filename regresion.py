import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(dataset):
    if dataset == '1':
        # Load the home rental dataset
        data = pd.read_csv('home-rental.csv')
    elif dataset == '2':
        # Load the ice cream dataset
        data = pd.read_csv('ice-cream.csv')
    else:
        raise ValueError("Invalid option. Select '1' for home-rental or '2' for ice-cream.")
    return data

def perform_regression(data, target_column=None, feature_columns=None):
    # automatically choose numeric columns if none provided
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Dataset does not contain any numeric columns for regression")

    # determine target column
    if target_column is None:
        # default to last numeric column
        target_column = numeric_cols[-1]
    if target_column not in numeric_cols:
        raise ValueError(f"Target column '{target_column}' is not numeric or not present in data")

    # determine feature columns
    if feature_columns is None:
        feature_columns = [c for c in numeric_cols if c != target_column]
    if not feature_columns:
        raise ValueError("No feature columns are available after excluding the target")

    X = data[feature_columns]
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return predictions, y_test, mae, rmse, r2

def visualize_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=predictions, scatter_kws={'color':'blue'}, line_kws={'color':'red'})
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Results')
    plt.show()

if __name__ == '__main__':
    print("Select Dataset:")
    print("1: home-rental.csv")
    print("2: ice-cream.csv")
    choice = input("Enter your choice (1 or 2): ")

    try:
        data = load_data(choice)
        # show what we've loaded
        print("\nColumns in dataset:", list(data.columns))
        numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        print("Numeric columns (candidates for regression):", numeric)

        # auto-pick target = last numeric
        target = None
        if numeric:
            target = numeric[-1]
            print(f"Using '{target}' as target column and the remaining numeric columns as features.")

        predictions, y_test, mae, rmse, r2 = perform_regression(data, target_column=target)

        print(f"\nMean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R² score: {r2}\n")

        visualize_results(y_test, predictions)
    except Exception as e:
        print(f"Error: {e}")
