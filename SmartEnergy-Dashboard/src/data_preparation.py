import pandas as pd
import numpy as np

def generate_and_clean_data():
    # --- 1. จำลองข้อมูล (Data Acquisition) ---
    date_range = pd.date_range(start='2024-01-01', end='2026-03-10', freq='H')
    df = pd.DataFrame({'timestamp': date_range})
    
    # จำลองอุณหภูมิภายนอก (อิงตามอากาศไทย 28-38 องศา)
    df['temp_outside'] = 30 + 5 * np.sin(df.index / 24) + np.random.normal(0, 1.5, len(df))
    
    # จำลองการเปิดแอร์ (เปิดช่วงเย็น 18:00 - 08:00 และถ้าอุณหภูมิ > 32)
    df['hour'] = df['timestamp'].dt.hour
    df['aircon_hours'] = np.where((df['temp_outside'] > 31) & ((df['hour'] >= 18) | (df['hour'] <= 8)), 1, 0)
    
    # Target: total_kwh (หน่วยไฟที่ใช้)
    df['total_kwh'] = 0.3 + (df['aircon_hours'] * 2.0) + (df['temp_outside'] * 0.01) + np.random.normal(0, 0.05, len(df))

    # --- 2. การทำความสะอาด (Data Cleaning - หัวใจสำคัญ!) ---
    # 2.1 ใส่ค่า Outliers แบบตั้งใจ (เพื่อให้มีงาน Clean)
    df.loc[100:105, 'total_kwh'] = 500.0  # ค่ารวนสูงผิดปกติ
    
    # 2.2 จัดการ Outliers (ใช้เทคนิค Clip หรือ Z-score)
    limit = df['total_kwh'].quantile(0.99)
    df['total_kwh'] = df['total_kwh'].clip(upper=limit)
    
    # 2.3 จัดการ Missing Values (สมมติข้อมูลหาย)
    df.loc[200:205, 'total_kwh'] = np.nan
    df['total_kwh'] = df['total_kwh'].fillna(method='ffill') # Forward Fill
    
    # 2.4 Feature Engineering (สร้างตัวแปรช่วยพยากรณ์)
    df['is_weekend'] = df['timestamp'].dt.dayofweek // 5
    df['month'] = df['timestamp'].dt.month
    
    df.to_csv('data/cleaned_energy_data.csv', index=False)
    print("Step 1: Data Prepared and Cleaned!")

if __name__ == "__main__":
    generate_and_clean_data()