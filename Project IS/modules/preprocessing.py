import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_data(filepath):
    """
    ฟังก์ชันสำหรับอ่านและเตรียมข้อมูล (Preprocessing)
    คืนค่า dict ที่มีข้อมูลรายละเอียดแต่ละขั้นตอนเพื่อนำไปแสดงผลบนเว็บไซต์
    """
    # 1. อ่านข้อมูล CSV
    df = pd.read_csv(filepath)
    
    # เก็บข้อมูลสรุปเพื่อแสดงให้ User เห็น
    steps_info = {}
    
    # ---------------------------------------------------------
    # Step 1: Raw Data Overview
    # ---------------------------------------------------------
    steps_info['step1_title'] = "1. ข้อมูลดิบ (Raw Data)"
    # แปลง dataframe 5 แถวแรกเป็น HTML table เพื่อแสดงผล
    steps_info['step1_data'] = df.head().to_html(classes='table table-striped table-hover', index=False)
    steps_info['step1_shape'] = f"จำนวนข้อมูล: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์"
    
    # ---------------------------------------------------------
    # Step 2: Missing Value Handling (การจัดการค่าว่าง)
    # ---------------------------------------------------------
    missing_before = df.isnull().sum().sum()
    
    # สำหรับข้อมูล Breast Cancer มักมีคอลัมน์ขยะเช่น 'Unnamed: 32' เราจะลบทิ้ง
    if 'Unnamed: 32' in df.columns:
        df = df.drop(['Unnamed: 32'], axis=1)
    
    # Drop N/A แถวที่มีค่าว่าง
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    
    steps_info['step2_title'] = "2. การจัดการค่าว่าง (Missing Value Handling)"
    steps_info['step2_desc'] = f"พบค่าว่างทั้งหมด: {missing_before} ค่า -> หลังจากทำการลบ (Drop N/A) ค่าว่างเหลือ: {missing_after} ค่า"
    
    # ---------------------------------------------------------
    # Step 3: Encoding Categorical Data (การแปลงข้อมูลตัวอักษร)
    # ---------------------------------------------------------
    # ลบ 'id' เพราะไม่มีผลต่อการทำนาย
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
        
    steps_info['step3_title'] = "3. การแปลงข้อมูลตัวอักษร (Label Encoding)"
    
    # ตรวจสอบว่ามีคอลัมน์ Diagnosis (เป้าหมาย) หรือไม่
    if 'diagnosis' in df.columns:
        le = LabelEncoder()
        # แปลง M (Malignant/เนื้อร้าย) และ B (Benign/เนื้อดี) เป็น (1,0)
        df['diagnosis'] = le.fit_transform(df['diagnosis'])
        steps_info['step3_desc'] = "แปลงคอลัมน์เป้าหมาย 'diagnosis' จากเนื้อร้าย(M)/เนื้อดี(B) เป็น 1 และ 0 ตามลำดับ"
    else:
         steps_info['step3_desc'] = "ไม่พบคอลัมน์ 'diagnosis' ข้ามขั้นตอนนี้"
    
    # ---------------------------------------------------------
    # Step 4: Feature Scaling (การปรับสเกลข้อมูลมาตรฐาน)
    # ---------------------------------------------------------
    steps_info['step4_title'] = "4. การปรับสเกลข้อมูล (Feature Scaling)"
    steps_info['step4_desc'] = "ใช้ StandardScaler ปรับค่าให้ Mean = 0 และ Standard Deviation = 1 เพื่อความแม่นยำของ NN และ SVM"
    
    # แยก X (Features) และ y (Target)
    if 'diagnosis' in df.columns:
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
    else:
        # หากไม่มี diagnosis ให้ถือว่าทุกคอลัมน์คือ x
        X = df
        y = None
        
    scaler = StandardScaler()
    # ปรับสเกลข้อมูลตัวเลขทั้งหมด
    X_scaled = scaler.fit_transform(X)
    
    # นำข้อมูลที่ปรับกลับมาใส่ DataFrame เพื่อให้ User มองเห็น
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    if y is not None:
        df_scaled['diagnosis'] = y.values
        
    # แสดง 5 แถวแรกหลังจากการ Scaling
    steps_info['step4_data'] = df_scaled.head().to_html(classes='table table-striped table-hover', index=False)
    
    # ส่งข้อมูล Dataset ที่ Preprocess เสร็จแล้ว (df_scaled) สำหรับใช้ Train โมเดลด้วย
    return steps_info, df_scaled
