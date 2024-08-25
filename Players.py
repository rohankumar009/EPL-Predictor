import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path): # stats loaded from CSV
    return pd.read_csv(file_path)


def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=['Player', 'PerformanceScore'])  # remving non-feature columns
    y = df['PerformanceScore'] # main target
    return X, y

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # standardized features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    return model, scaler


def visualize_performance(model, X, y):
    y_pred = model.predict(X)
    
    plt.scatter(y, y_pred, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Performance Score')
    plt.ylabel('Predicted Performance Score')
    plt.title('Player Performance Prediction')
    plt.show()

if __name__ == "__main__":
    data = load_data('player_stats.csv')

    X, y = preprocess_data(data)

    model, scaler = train_model(X, y)
    
    X_scaled = scaler.transform(X)
    visualize_performance(model, X_scaled, y)
