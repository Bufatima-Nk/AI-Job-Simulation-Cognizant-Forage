import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

class StockPredictionModel:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_clean_data(self, filename):
        df = pd.read_csv(filename)
        df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
        return df

    def preprocess_data(self, sales_file, stock_file, temp_file):
        sales_df = self.load_and_clean_data(sales_file)
        stock_df = self.load_and_clean_data(stock_file)
        temp_df = self.load_and_clean_data(temp_file)
        
        # Convert timestamps to hourly format
        timestamp_columns = ['timestamp']
        for df in [sales_df, stock_df, temp_df]:
            for col in timestamp_columns:
                df = self.convert_timestamp_to_hourly(df, col)

        # Merge dataframes
        sales_agg = sales_df.groupby(['timestamp', 'product_id']).agg({'quantity': 'sum'}).reset_index()
        stock_agg = stock_df.groupby(['timestamp', 'product_id']).agg({'estimated_stock_pct': 'mean'}).reset_index()
        temp_agg = temp_df.groupby(['timestamp']).agg({'temperature': 'mean'}).reset_index()

        merged_df = stock_agg.merge(sales_agg, on=['timestamp', 'product_id'], how='left')
        merged_df = merged_df.merge(temp_agg, on='timestamp', how='left')
        merged_df = merged_df.merge(product_categories, on="product_id", how="left")
        merged_df = merged_df.merge(product_price, on="product_id", how="left")

        # Handle missing values and other preprocessing steps
        merged_df.fillna(0, inplace=True)  # Filling missing values with zeros

        # Split features and target
        X = merged_df.drop(columns=['estimated_stock_pct'])
        y = merged_df['estimated_stock_pct']

        return merged_df, X, y

    def train_model(self, X, y):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        return best_model

    def evaluate_model(self, model, X_test, y_test):
        test_predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, test_predictions, squared=False)
        mae = mean_absolute_error(y_test, test_predictions)
        return rmse, mae

    def save_model(self, model, filename):
        joblib.dump(model, filename)

if __name__ == "__main__":
    model = StockPredictionModel()

    merged_df, X, y = model.preprocess_data("sales.csv", "sensor_stock_levels.csv", "sensor_storage_temperature.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = model.train_model(X_train, y_train)
    rmse, mae = model.evaluate_model(best_model, X_test, y_test)

    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae.round(2))

    model.save_model(best_model, "stock_prediction_model.pkl")
