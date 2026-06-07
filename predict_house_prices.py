import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("      SkillCraft Technology - Machine Learning Internship")
    print("                  TASK 01: House Price Predictor")
    print("=" * 60)

    # 1. Load the dataset
    data_path = os.path.join(os.path.dirname(__file__), "house_prices.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"\n[+] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Exploratory Data Analysis (EDA)
    print("\n--- Exploratory Data Analysis ---")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Summary Statistics:")
    print(df.describe())
    
    print("\nDataset Information:")
    print(df.info())

    # 3. Splitting Features and Target Variable
    # Features: SquareFootage, Bedrooms, Bathrooms
    # Target: Price
    X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
    y = df['Price']

    # 4. Train Test Split
    print("\n[+] Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Build and Train the Model
    print("[+] Initializing and training the Multiple Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Model Parameters
    print("\n--- Model Coefficients & Parameters ---")
    print(f"Intercept (Base Price): ${model.intercept_:,.2f}")
    features = list(X.columns)
    for idx, col in enumerate(features):
        print(f"Coefficient for {col}: {model.coef_[idx]:,.2f}")

    # 7. Model Evaluation
    y_pred = model.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Metrics ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared Score (R²): {r2:.4f} ({r2*100:.2f}% variance explained)")

    # 8. Generate and Save Visualizations
    print("\n[+] Saving visualization plots...")
    plt.figure(figsize=(10, 6))
    
    # Plotting actual vs predicted
    plt.scatter(y_test, y_pred, color='teal', edgecolors='black', alpha=0.7, label='Data Points')
    # Perfect fit line
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Fit Line')
    
    plt.title('Actual vs Predicted House Prices (Linear Regression)', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Prices ($)', fontsize=12)
    plt.ylabel('Predicted Prices ($)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plot_path = os.path.join(os.path.dirname(__file__), "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved visualization plot to: {plot_path}")

    # 9. Interactive prediction for users
    print("\n" + "=" * 60)
    print("               Interactive Prediction System")
    print("=" * 60)
    print("Enter the details of a house below to predict its market price.")
    
    while True:
        try:
            sqft_input = input("\nEnter Square Footage (e.g. 1500) [or 'q' to quit]: ").strip()
            if sqft_input.lower() == 'q':
                print("Exiting prediction system.")
                break
            
            sqft = float(sqft_input)
            bedrooms = float(input("Enter number of Bedrooms (e.g. 3): "))
            bathrooms = float(input("Enter number of Bathrooms (e.g. 2): "))
            
            # Predict
            user_features = np.array([[sqft, bedrooms, bathrooms]])
            predicted_price = model.predict(user_features)[0]
            
            print("-" * 45)
            print(f"Predicted House Price: ${predicted_price:,.2f}")
            print("-" * 45)
            
        except ValueError:
            print("Invalid input! Please enter numeric values or 'q' to exit.")
        except KeyboardInterrupt:
            print("\nExiting prediction system.")
            break

if __name__ == "__main__":
    main()
