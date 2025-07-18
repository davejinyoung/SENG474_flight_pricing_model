import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class EnhancedAirlineFarePredictor:
    def __init__(self):
        self.sklearn_model = None
        self.manual_model = None
        self.scaler = None
        self.selected_features = None
        self.feature_importances = None
        self.is_trained = False
        self.model_metrics = {}
        self.theta = None
        
    def load_and_prepare_data(self, df):
        # Load and prepare the airline data
        print("Preparing airline fare data...")
        print(f"Original dataset shape: {df.shape}")
        
        # Remove non-numeric columns that might cause issues
        df = df.select_dtypes(include=['number'])
        
        print(f"Dataset shape after cleaning: {df.shape}")
        print(f"Fare range: ${df['Average_Fare'].min():.2f} - ${df['Average_Fare'].max():.2f}")
        print(f"Mean fare: ${df['Average_Fare'].mean():.2f}")
        
        return df
    
    def forward_feature_selection(self, X, y, max_features=5):
        # Forward feature selection using MSE as the criterion
        # Returns top 5 most impactful features
    
        print("\nPerforming forward feature selection...")
        
        best_features = []
        best_score = np.inf  # Initialize for minimization
        features = list(X.columns)
        
        while features and len(best_features) < max_features:
            mse_scores = {}
            
            for feature in features:
                selected = best_features + [feature]
                new_X = X[selected]
                
                # Split data for evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    new_X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model using normal equation
                theta = self.linear_regression_normal_equation(X_train_scaled, y_train.values)
                
                # Make predictions
                y_pred = self.predict_with_theta(X_test_scaled, theta)
                mse = mean_squared_error(y_test, y_pred)
                
                mse_scores[feature] = mse
            
            # Find the best feature to add
            best_feature, new_best_score = min(mse_scores.items(), key=lambda item: item[1])
            
            # Add feature if it improves the score significantly
            if new_best_score < best_score - 10:
                best_features.append(best_feature)
                features.remove(best_feature)
                improvement = abs(new_best_score - best_score) if not np.isinf(best_score) else new_best_score
                print(f"Selected: {best_feature} | MSE: {new_best_score:.3f} | Improvement: {improvement:.3f}")
                best_score = new_best_score
            else:
                break
        
        print(f"\nFinal selected features: {best_features}")
        return best_features
    
    def linear_regression_normal_equation(self, X, y):
        # Implement linear regression using the normal equation:
        # θ = (X^T X)^(-1) X^T y

        # Add bias term (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calc the coefficients using the normal equation
        try:
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            # (if matrix is singular, use pseudo-inverse)
            theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        return theta
    
    def predict_with_theta(self, X, theta):
        # Make predictions using the trained model parameters 
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(theta)
    
    def plot_target_distribution(self, y):
        # Plot the distribution of the target variable 
        print("\nAnalyzing target variable distribution...")
        
        data = np.array(y)
        mean_value = np.mean(data)
        std_dev = np.std(data)
        
        print(f"Mean fare price: ${mean_value:.2f}")
        print(f"Standard deviation: ${std_dev:.2f}")
        
        # Create normal distribution curve
        x = np.linspace(mean_value - 4*std_dev, mean_value + 4*std_dev, 500)
        pdf = norm.pdf(x, loc=mean_value, scale=std_dev)
        
        plt.figure(figsize=(12, 6))
        
        # histogram and normal distribution
        plt.subplot(1, 2, 1)
        plt.plot(x, pdf, color='blue', linewidth=2, label='Normal Distribution PDF')
        plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Histogram of Fares')
        plt.title('Target Variable Distribution')
        plt.xlabel('Fare Price ($)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        
        # boxplot
        plt.subplot(1, 2, 2)
        plt.boxplot(data, vert=True)
        plt.title('Fare Price Box Plot')
        plt.ylabel('Fare Price ($)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, y_true, y_pred, model_name):
        # Evaluate model performance using multiple metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"  Mean Squared Error: {mse:.2f}")
        print(f"  Root Mean Squared Error: ${rmse:.2f}")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'predictions': y_pred
        }
    
    def train_model(self, df):
        #Train the linear regression model
        print("\nTraining linear regression model...")
        print("="*60)
        
        # Prepare data
        df_clean = self.load_and_prepare_data(df)
        
        # Separate features and target
        X = df_clean.drop('Average_Fare', axis=1)
        y = df_clean['Average_Fare']
        
        # plot target distribution
        self.plot_target_distribution(y)
        
        # Perform forward feature selection to get top 5 features
        self.selected_features = self.forward_feature_selection(X, y, max_features=5)
        
        # Use selected features
        X_selected = X[self.selected_features]
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # 1. Scikit-learn model (for comparison)
        print("\nTraining Scikit-learn model...")
        self.sklearn_model = LinearRegression()
        self.sklearn_model.fit(X_train_scaled, y_train)
        sklearn_pred = self.sklearn_model.predict(X_test_scaled)
        
        # 2. Manual implementation using normal equation
        print("Training manual model using normal equation...")
        self.theta = self.linear_regression_normal_equation(X_train_scaled, y_train.values)
        manual_pred = self.predict_with_theta(X_test_scaled, self.theta)
        
        # Evaluate both models
        sklearn_metrics = self.evaluate_model(y_test, sklearn_pred, "Scikit-learn Model")
        manual_metrics = self.evaluate_model(y_test, manual_pred, "Manual Normal Equation Model")
        
        # Store metrics
        self.model_metrics = {
            'Scikit-learn': sklearn_metrics,
            'Manual': manual_metrics
        }
        
        # Compare coefficients
        print(f"\n{'='*60}")
        print("COEFFICIENT COMPARISON")
        print(f"{'='*60}")
        print(f"{'Feature':<25} {'Scikit-learn':<15} {'Manual':<15}")
        print("-" * 55)
        
        for i, feature in enumerate(self.selected_features):
            sklearn_coef = self.sklearn_model.coef_[i]
            manual_coef = self.theta[i + 1]  # +1 because of bias term
            print(f"{feature:<25} {sklearn_coef:<15.4f} {manual_coef:<15.4f}")
        
        # Print intercepts
        print(f"{'Intercept':<25} {self.sklearn_model.intercept_:<15.4f} {self.theta[0]:<15.4f}")
        
        # Feature importance analysis
        print(f"\n{'='*60}")
        print("FEATURE ANALYSIS")
        print(f"{'='*60}")
        
        # Calculate feature importances based on absolute coefficient values
        feature_importance = {}
        for i, feature in enumerate(self.selected_features):
            importance = abs(self.sklearn_model.coef_[i])
            feature_importance[feature] = importance
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Feature Importance (based on absolute coefficient values):")
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        self.feature_importances = dict(sorted_features)
        self.is_trained = True
        
        return self.model_metrics
    
    def get_user_input(self):
        """Get the feature values from user input for the top 5 features"""
        print("\n" + "="*60)
        print("ENTER FLIGHT DETAILS FOR FARE PREDICTION")
        print("="*60)
        
        # map selected features to user-friendly prompts
        feature_prompts = {
            'NonStopMiles': 'Flight distance in miles (e.g., 1500)',
            'MktMilesFlown': 'Total market miles flown including connections (e.g., 1600)',
            'Market_HHI': 'Market concentration index (1000-10000, higher = less competition, e.g., 3000)',
            'Market_share': 'Market share (0.0 to 1.0, e.g., 0.3 for 30%)',
            'MktCoupons': 'Number of market segments/coupons (1 for direct, 2 for 1-stop, etc.)',
            'Pax': 'Number of passengers (e.g., 150)',
            'LCC_Comp': 'Low-cost carrier competition (0 or 1)',
            'Circuity': 'Route circuity factor (1.0 for direct, higher for indirect)',
            'Carrier_freq': 'Carrier frequency (e.g., 0.1)',
            'ODPairID_freq': 'Origin-Destination pair frequency (e.g., 0.0001)'
        }
        
        input_dict = {}
        
        print("Please enter the following flight details:")
        print("-" * 50)
        
        # Get input for the selected features
        for feature in self.selected_features:
            prompt = feature_prompts.get(feature, f'{feature} value')
            while True:
                try:
                    value = float(input(f"{prompt}: "))
                    
                    # Basic validation
                    if feature == 'NonStopMiles' and (value < 50 or value > 6000):
                        print("Flight distance should be between 50 and 6000 miles")
                        continue
                    if feature == 'Market_share' and (value < 0 or value > 1):
                        print("Market share must be between 0 and 1")
                        continue
                    if feature == 'MktCoupons' and (value < 1 or value > 10):
                        print("Market coupons must be between 1 and 10")
                        continue
                    if feature == 'LCC_Comp' and value not in [0, 1]:
                        print("LCC competition must be 0 or 1")
                        continue
                    
                    input_dict[feature] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Prepare feature values in the correct order
        feature_values = [input_dict[feature] for feature in self.selected_features]
        
        return feature_values
    
    def predict_fare(self, feature_values):
        # Make fare prediction using both models

        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Convert to numpy array and scale
        input_data = np.array(feature_values).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        
        # Make predictions
        sklearn_pred = self.sklearn_model.predict(input_scaled)[0]
        manual_pred = self.predict_with_theta(input_scaled, self.theta)[0]
        
        # Ensure reasonable minimum fare
        sklearn_pred = max(50, sklearn_pred)
        manual_pred = max(50, manual_pred)
        
        predictions = {
            'Scikit-learn': sklearn_pred,
            'Manual (Normal Equation)': manual_pred
        }
        
        return predictions
    
    def interactive_prediction(self):
        # Interactive prediction interface

        if not self.is_trained:
            print("Error: Model not trained yet!")
            return
        
        print("\n" + "="*60)
        print("AIRLINE FARE PREDICTION SYSTEM")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Make a fare prediction")
            print("2. View model information")
            print("3. Show feature importance")
            print("4. Exit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                try:
                    feature_values = self.get_user_input()
                    predictions = self.predict_fare(feature_values)
                    
                    print(f"\n{'='*60}")
                    print("PREDICTION RESULTS")
                    print(f"{'='*60}")
                    
                    for model_name, prediction in predictions.items():
                        print(f"{model_name}: ${prediction:.2f}")
                    
                    # Calculate average
                    avg_prediction = np.mean(list(predictions.values()))
                    print(f"\nAverage Prediction: ${avg_prediction:.2f}")
                    print(f"{'='*60}")
                    
                except Exception as e:
                    print(f"Error making prediction: {e}")
            
            elif choice == '2':
                print(f"\n{'='*60}")
                print("MODEL INFORMATION")
                print(f"{'='*60}")
                
                for model_name, metrics in self.model_metrics.items():
                    print(f"\n{model_name}:")
                    print(f"  R² Score: {metrics['R2']:.4f}")
                    print(f"  RMSE: ${metrics['RMSE']:.2f}")
                    print(f"  MAE: ${metrics['MAE']:.2f}")
                    print(f"  MSE: {metrics['MSE']:.2f}")
                
                print(f"\nSelected Features ({len(self.selected_features)}):")
                for i, feature in enumerate(self.selected_features, 1):
                    print(f"  {i}. {feature}")
                print(f"{'='*60}")
            
            elif choice == '3':
                print(f"\n{'='*60}")
                print("FEATURE IMPORTANCE")
                print(f"{'='*60}")
                
                if self.feature_importances:
                    for i, (feature, importance) in enumerate(self.feature_importances.items(), 1):
                        print(f"  {i}. {feature}: {importance:.4f}")
                else:
                    print("No feature importance data available.")
                print(f"{'='*60}")
            
            elif choice == '4':
                print("Thank you for using the Airline Fare Prediction System!")
                break
            
            else:
                print("Invalid choice. Please select 1-4.")


def main():
    # Main function to load real CSV data and train the model
    
    print("AIRLINE FARE PREDICTION SYSTEM")
    print("="*60)
    print("Loading real airline data from CSV file...")
    
    # Load the CSV file
    try:
        # replace with your actual CSV file path
        csv_filename = "data/MarketFarePredictionData.csv"
        
        # Load the data
        df = pd.read_csv(csv_filename)
        
        print(f"Successfully loaded dataset!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show basic statistics
        print(f"\nDataset Statistics:")
        print(f"Average Fare range: ${df['Average_Fare'].min():.2f} - ${df['Average_Fare'].max():.2f}")
        print(f"Mean fare: ${df['Average_Fare'].mean():.2f}")
        print(f"Number of records: {len(df)}")
        
        # Initialize and train predictor
        predictor = EnhancedAirlineFarePredictor()
        predictor.train_model(df)
        
        # Start interactive prediction
        predictor.interactive_prediction()
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file '{csv_filename}'.")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your CSV file format and try again.")


if __name__ == "__main__":
    main()