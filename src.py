# =========================================================
# 📘 CSV REQUIREMENTS & DATA FORMAT
# =========================================================
"""
CSV FILE REQUIREMENTS
---------------------
Your CSV should look like:

order_placed_at,burger,pizza,coke
2025-10-25 10:05:00,2,4,1
2025-10-25 11:05:00,0,7,4
...

✅ REQUIRED:
- 'order_placed_at' : Timestamp of the order
- Columns for each dish with quantities (e.g., burger, pizza, coke)

⚙️ INTERNAL FEATURE ENGINEERING:
- hour : Extracted from 'order_placed_at'
- day_of_week : 0=Monday, 6=Sunday
- is_weekend : 1 if Saturday/Sunday
- lag features : Previous 1–3 hours per dish

🔢 DATA REQUIREMENT:
- Minimum: 200+ rows
- Recommended: 500–1000+ rows for stable results
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import random


# =========================================================
# 🧠 FUNCTION 1: Train Multi-Dish Demand Model
# =========================================================
def train_demand_model(csv_path: str, save_path: str = "trained_model.pkl"):
    df = pd.read_csv(csv_path, parse_dates=['order_placed_at'])
    
    # Time features
    df['hour'] = df['order_placed_at'].dt.hour
    df['day_of_week'] = df['order_placed_at'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Identify dish columns
    dish_cols = [c for c in df.columns if c not in ['order_placed_at', 'hour', 'day_of_week', 'is_weekend']]
    
    # Sort by timestamp
    df = df.sort_values('order_placed_at').reset_index(drop=True)
    
    # Lag features: previous 1,2,3 hours per dish
    lag_hours = [1,2,3]
    for dish in dish_cols:
        for lag in lag_hours:
            df[f'{dish}_lag{lag}'] = df[dish].shift(lag).fillna(0)
    
    # Feature matrix
    feature_cols = ['hour', 'day_of_week', 'is_weekend'] + [c for c in df.columns if '_lag' in c]
    X = df[feature_cols]
    y = df[dish_cols]
    
    # Train/test split (last 20% for validation)
    split_idx = int(len(df)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train CatBoost MultiOutputRegressor
    model = MultiOutputRegressor(
        CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.1,
            verbose=0,
            random_state=42
        )
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2_scores = {dish: r2_score(y_test[dish], y_pred[:,i]) for i, dish in enumerate(dish_cols)}
    print("✅ Model trained | Per-dish R² scores:", r2_scores)
    
    # Save model
    joblib.dump(model, save_path)
    print(f"💾 Model saved to {save_path}")
    
    return model, df, feature_cols, dish_cols

# =========================================================
# 🔮 FUNCTION 2: Predict Next Hour Demand
# =========================================================
def predict_next_hour(model, df, feature_cols, dish_cols):
    last_time = df['order_placed_at'].max()
    next_hour = last_time + pd.Timedelta(hours=1)
    
    # Build features
    hour = next_hour.hour
    day_of_week = next_hour.dayofweek
    is_weekend = int(day_of_week in [5,6])
    
    # Lag values: take last row's dish counts
    last_row = df.iloc[-1]
    feature_dict = {'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend}
    for dish in dish_cols:
        feature_dict[f'{dish}_lag1'] = last_row[dish]
        feature_dict[f'{dish}_lag2'] = df[dish].iloc[-2] if len(df) > 1 else 0
        feature_dict[f'{dish}_lag3'] = df[dish].iloc[-3] if len(df) > 2 else 0
    
    X_next = pd.DataFrame([feature_dict])
    pred = model.predict(X_next)[0]
    
    result = dict(zip(dish_cols, pred))
    print(f"🕐 Predicted dishes for next hour ({next_hour}): {result}")
    return result

# =========================================================
# 🔮 FUNCTION 3: Predict Next 24 Hours
# =========================================================
def predict_next_24_hours(model, df, feature_cols, dish_cols):
    df_copy = df.copy().sort_values('order_placed_at').reset_index(drop=True)
    last_time = df_copy['order_placed_at'].max()
    
    predictions = []
    
    # Initialize rolling dish history for lag features
    rolling_history = {dish: df_copy[dish].tolist() for dish in dish_cols}
    
    for i in range(1, 25):  # next 24 hours
        next_hour = last_time + pd.Timedelta(hours=i)
        hour = next_hour.hour
        day_of_week = next_hour.dayofweek
        is_weekend = int(day_of_week in [5,6])
        
        # Build lag features
        feature_dict = {'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend}
        for dish in dish_cols:
            # get last 3 hours for this dish (fill 0 if not enough history)
            feature_dict[f'{dish}_lag1'] = rolling_history[dish][-1] if len(rolling_history[dish]) >= 1 else 0
            feature_dict[f'{dish}_lag2'] = rolling_history[dish][-2] if len(rolling_history[dish]) >= 2 else 0
            feature_dict[f'{dish}_lag3'] = rolling_history[dish][-3] if len(rolling_history[dish]) >= 3 else 0
        
        X_next = pd.DataFrame([feature_dict])
        pred = model.predict(X_next)[0]
        
        # Append prediction
        predictions.append({'timestamp': next_hour, **dict(zip(dish_cols, pred))})
        
        # Update rolling history
        for j, dish in enumerate(dish_cols):
            rolling_history[dish].append(pred[j])
    
    pred_df = pd.DataFrame(predictions)
    print("🌤 Next 24-hour demand forecast generated.")
    return pred_df



def generate_realistic_hourly_orders(target_rows=1008):
    """
    Generate realistic hourly-aggregated order data:
    - Randomly select 4–8 dishes from a larger pool
    - Randomize base popularities for each selected dish (1–5)
    - Controlled zeros, peak hour boosts, weekend boost, small spikes and rare outliers
    - Returns a DataFrame with 'order_placed_at' and one column per selected dish
    """
    # Larger pool of possible dishes
    all_dishes = [
        "Burger", "Pizza", "Coke", "Fries", "Salad",
        "Pasta", "IceCream", "Sushi", "Sandwich", "Soup",
        "Wrap", "Taco", "Nuggets", "Donut", "Tea", "Smoothie"
    ]
    
    # Choose random number of dishes between 4 and 8
    num_dishes = random.randint(4, 8)
    dishes = random.sample(all_dishes, k=num_dishes)
    
    # Randomize base popularities (1 to 5) for each selected dish
    base_popularity = list(np.random.randint(1, 6, size=len(dishes)))
    
    # Hourly timestamps ending now, respecting target_rows
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(end=end_time, periods=target_rows, freq='H')
    
    data = []
    
    for ts in timestamps:
        row = []
        hour = ts.hour
        weekday = ts.weekday()
        weekend_factor = 1.2 if weekday >= 5 else 1.0  # 20% boost weekends
        
        for idx, dish in enumerate(dishes):
            popularity = base_popularity[idx]
            
            # Peak hour boost
            if 11 <= hour <= 14 or 18 <= hour <= 21:
                popularity_effective = popularity * 1.5
            else:
                popularity_effective = popularity
            
            # Off-peak reduction
            if hour < 8 or hour > 22:
                popularity_effective *= 0.5  # reduce lambda, less orders
            
            popularity_effective *= weekend_factor
            
            # Sample from Poisson (ensures mostly realistic counts)
            qty = np.random.poisson(lam=max(0.1, popularity_effective))
            
            # Small spike: 10% chance
            if random.random() < 0.1:
                qty += random.randint(1, 3)
            
            # Rare additive outlier: 2% chance
            if random.random() < 0.02:
                qty += random.randint(5, 10)
            
            # Cap maximum per dish
            qty = min(qty, 20)
            
            row.append(qty)
        data.append(row)
    
    df = pd.DataFrame(data, columns=dishes)
    df.insert(0, "order_placed_at", timestamps)
    
    return df

