import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class AirlinePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['NonStopMiles', 'MktMilesFlown', 'Market_HHI', 'Market_share', 'MktCoupons']
        self.target = 'Average_Fare'
        self.best_k = None
        
    def load_and_train(self, csv_file='data/MarketFarePredictionData.csv'):
        """Load data and train the KNN model"""
        # --- 1. Load Data ---
        try:
            df = pd.read_csv(csv_file)
            print("Dataset loaded successfully.")
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            print(f"Error: '{csv_file}' not found. Please ensure the file is in the current directory.")
            return False

        # --- 2. Define Features and Target ---
        X = df[self.features]
        y = df[self.target]
        
        print(f"\nFeature statistics:")
        print(X.describe())

        # --- 3. Create the Three-Way Split ---
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        print(f"\nData split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples.")

        # --- 4. Scale the Features ---
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # --- 5. Find the Optimal 'k' using the Validation Set ---
        print("\nFinding optimal k value...")
        k_range = range(1, 41)
        rmse_values = []

        for k in k_range:
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            error = np.sqrt(mean_squared_error(y_val, y_val_pred))
            rmse_values.append(error)

        # Find the best k value
        self.best_k = k_range[np.argmin(rmse_values)]
        print(f"Optimal k found: {self.best_k}")

        # Plot the RMSE vs. K value graph
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, rmse_values, marker='o', linestyle='--')
        plt.title('RMSE vs. K Value (on Validation Set)')
        plt.xlabel('K Value')
        plt.ylabel('Root Mean Squared Error (RMSE)')
        plt.axvline(self.best_k, color='r', linestyle='--', label=f'Best k = {self.best_k}')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- 6. Train the Final Model ---
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_full)
        X_train_full_scaled = self.scaler.transform(X_train_full)
        X_test_final_scaled = self.scaler.transform(X_test)

        # Train the final model
        start_train_time = time.time()
        self.model = KNeighborsRegressor(n_neighbors=self.best_k)
        self.model.fit(X_train_full_scaled, y_train_full)
        end_train_time = time.time()
        training_time = end_train_time - start_train_time

        # Make final predictions on the test set
        start_pred_time = time.time()
        y_test_pred = self.model.predict(X_test_final_scaled)
        end_pred_time = time.time()
        prediction_time_total = end_pred_time - start_pred_time

        # --- 7. Report Final Performance Metrics ---
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        prediction_time_per_1k = (prediction_time_total / len(X_test)) * 1000

        print("\n" + "="*60)
        print("FINAL MODEL PERFORMANCE ON TEST SET")
        print("="*60)
        print(f"Model: KNN Regressor with k={self.best_k}")
        print(f"Accuracy Metrics:")
        print(f"  • Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"  • Mean Absolute Error (MAE):      ${mae:.2f}")
        print(f"  • R-squared (R²):                 {r2:.3f}")
        print(f"\nEfficiency Metrics:")
        print(f"  • Training Time:                  {training_time:.6f} seconds")
        print(f"  • Prediction Time (per 1k rows):  {prediction_time_per_1k:.6f} ms")
        print("="*60)

        # --- 8. Visualize Results ---
        self._plot_results(y_test, y_test_pred)
        
        # Save the model and scaler
        self._save_model()
        
        return True
    
    def _plot_results(self, y_test, y_test_pred):
        """Create visualization plots"""
        # Actual vs. Predicted Fare Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
        p1 = max(max(y_test_pred), max(y_test))
        p2 = min(min(y_test_pred), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'r--')
        plt.xlabel('Actual Fare ($)')
        plt.ylabel('Predicted Fare ($)')
        plt.title(f'Actual vs. Predicted Fares (k={self.best_k})')
        plt.grid(True)

        # Residuals Plot
        residuals = y_test - y_test_pred
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Fare ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residuals vs. Predicted Fares (k={self.best_k})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            joblib.dump(self.model, 'knn_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            print("\nModel and scaler saved successfully!")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            self.model = joblib.load('knn_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("Pre-trained model loaded successfully!")
            return True
        except Exception as e:
            print(f"Could not load pre-trained model: {e}")
            return False
    
    def predict_price(self, non_stop_miles, mkt_miles_flown, market_hhi, market_share, mkt_coupons):
        """Make a price prediction for given inputs"""
        if self.model is None or self.scaler is None:
            print("Error: Model not trained or loaded. Please train the model first.")
            return None
        
        # Create input array
        input_data = np.array([[non_stop_miles, mkt_miles_flown, market_hhi, market_share, mkt_coupons]])
        
        # Scale the input
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        return prediction
    
    def interactive_prediction(self):
        """Interactive interface for making predictions"""
        if self.model is None or self.scaler is None:
            print("Error: Model not trained or loaded. Please train the model first.")
            return
        
        print("\n" + "="*60)
        print("AIRLINE TICKET PRICE PREDICTOR")
        print("="*60)
        print("Enter the following flight characteristics to get a price prediction:")
        print()
        
        try:
            # Get user input for each feature
            print("1. Non-Stop Miles:")
            print("   (Direct distance between origin and destination airports)")
            non_stop_miles = float(input("   Enter value: "))
            
            print("\n2. Market Miles Flown:")
            print("   (Total miles flown in this market route)")
            mkt_miles_flown = float(input("   Enter value: "))
            
            print("\n3. Market HHI (Herfindahl-Hirschman Index):")
            print("   (Market concentration measure, typically 0-10000)")
            market_hhi = float(input("   Enter value: "))
            
            print("\n4. Market Share:")
            print("   (Carrier's market share as a decimal, e.g., 0.25 for 25%)")
            market_share = float(input("   Enter value: "))
            
            print("\n5. Market Coupons:")
            print("   (Number of flight segments, typically 1 for non-stop, 2+ for connecting)")
            mkt_coupons = float(input("   Enter value: "))
            
            # Make prediction
            predicted_price = self.predict_price(non_stop_miles, mkt_miles_flown, 
                                               market_hhi, market_share, mkt_coupons)
            
            # Display results
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print("Input Values:")
            print(f"  • Non-Stop Miles:     {non_stop_miles:,.0f}")
            print(f"  • Market Miles Flown: {mkt_miles_flown:,.0f}")
            print(f"  • Market HHI:         {market_hhi:,.0f}")
            print(f"  • Market Share:       {market_share:.3f} ({market_share*100:.1f}%)")
            print(f"  • Market Coupons:     {mkt_coupons:.0f}")
            print(f"\nPredicted Ticket Price: ${predicted_price:.2f}")
            print("="*60)
            
        except ValueError:
            print("Error: Please enter valid numeric values.")
        except KeyboardInterrupt:
            print("\nPrediction cancelled.")
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    """Main function to run the airline price predictor"""
    predictor = AirlinePricePredictor()
    
    print("Airline Ticket Price Predictor using K-Nearest Neighbors")
    print("="*60)
    
    # Try to load existing model first
    if os.path.exists('knn_model.pkl') and os.path.exists('scaler.pkl'):
        print("Found existing trained model.")
        choice = input("Do you want to (1) use existing model or (2) retrain? Enter 1 or 2: ")
        
        if choice == '1':
            if predictor.load_model():
                predictor.interactive_prediction()
                return
            else:
                print("Failed to load existing model. Will retrain...")
    
    # Train new model
    print("Training new model...")
    if predictor.load_and_train():
        print("\nModel training completed!")
        
        # Ask if user wants to make predictions
        while True:
            choice = input("\nWould you like to make a price prediction? (y/n): ").lower()
            if choice == 'y':
                predictor.interactive_prediction()
                
                # Ask if they want to make another prediction
                another = input("\nMake another prediction? (y/n): ").lower()
                if another != 'y':
                    break
            elif choice == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
    else:
        print("Failed to train model. Please check your data file.")

if __name__ == "__main__":
    main()