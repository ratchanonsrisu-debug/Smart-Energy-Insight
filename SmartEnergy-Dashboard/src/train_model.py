from pycaret.regression import *
import pandas as pd

def train_energy_model():
    df = pd.read_csv('data/cleaned_energy_data.csv')
    
    # เลือก Features ที่จะใช้ (ตัด timestamp ออกเพราะ PyCaret จะแยกส่วนประกอบให้เอง)
    # เราจะพยากรณ์ total_kwh
    s = setup(data=df, target='total_kwh', 
              ignore_features=['timestamp'], 
              session_id=296) # ใช้เลข Student ID ของคุณเป็น Seed
    
    # เปรียบเทียบ Model (โชว์ในรายงานได้)
    best = compare_models()
    
    # บันทึก Model ไว้ใช้ใน Dash
    save_model(best, 'models/best_energy_model')
    print("Step 2: Model Trained and Saved!")

if __name__ == "__main__":
    train_energy_model()