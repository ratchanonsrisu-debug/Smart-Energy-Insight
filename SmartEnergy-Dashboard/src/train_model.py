import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_energy_model():
    # 1. โหลดข้อมูล
    df = pd.read_csv('data/cleaned_energy_data.csv')
    
    # 2. เลือก Features (X) และ Target (y)
    # เราจะใช้ temp_outside, aircon_hours, is_weekend, month เป็นตัวพยากรณ์
    features = ['temp_outside', 'aircon_hours', 'is_weekend', 'month']
    X = df[features]
    y = df['total_kwh']
    
    # 3. แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=296)
    
    # 4. สร้างและเทรน Model (ใช้ RandomForest เหมือนที่ PyCaret มักจะเลือกให้)
    model = RandomForestRegressor(n_estimators=100, random_state=296)
    model.fit(X_train, y_train)
    
    # 5. วัดผล (เก็บค่านี้ไว้ใส่ในรายงานด้วยนะ)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Trained! Mean Absolute Error: {mae:.4f}")
    
    # 6. บันทึก Model ไว้ใช้ใน Dash (ไฟล์ .pkl)
    joblib.dump(model, 'models/best_energy_model.pkl')
    print("Step 2: Model Saved as models/best_energy_model.pkl")

if __name__ == "__main__":
    train_energy_model()