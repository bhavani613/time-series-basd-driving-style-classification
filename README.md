# 🚗 Driving Style Detection System

## 📌 Project Overview
This project is an AI-based Driving Style Detection System developed using Python and Streamlit.  
The system analyzes driving parameters such as time, speed, acceleration, brake usage, and steering angle to classify driving behavior into three categories: Normal, Drowsy, or Aggressive.

The application provides an interactive dashboard where users can upload a dataset, manually enter driving values, or upload a driving video to analyze driver behavior.

---

## ✨ Features
- Upload driving dataset (CSV)
- Manual driving parameter input
- Driving video upload and analysis
- Detection of Normal, Drowsy, and Aggressive driving styles
- Data visualization using graphs and pie charts
- Dashboard-style user interface
- Brake status detection (Applied / Not Applied)

---

## 🧠 Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- OpenCV

---

## 📊 Input Parameters
The system uses the following parameters to analyze driving behavior:

- Time
- Speed
- Acceleration
- Brake
- Steering Angle

---

## 🎯 Output
The system classifies driving behavior into:

- 🟢 Normal Driving – Safe and stable driving
- 🟠 Drowsy Driving – Possible driver fatigue
- 🔴 Aggressive Driving – Risky driving behavior

---

## 🚀 How to Run the Project

1. Install required libraries

pip install streamlit pandas numpy plotly opencv-python

2. Run the application

streamlit run app.py

3. Open the browser and interact with the dashboard.

---

## 📁 Project Structure

Driving-Style-Detection  
│  
├── app.py  
├── dataset_without_style.csv  
├── README.md  

---

## 📌 Purpose
This project demonstrates how data analysis and AI techniques can be used to monitor driver behavior and improve road safety by identifying risky driving patterns.