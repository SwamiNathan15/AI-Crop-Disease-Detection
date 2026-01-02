# 🌱 AI Crop Disease Detection

A **Flask-based Deep Learning web application** that detects **tomato crop diseases** from leaf images using a trained **TensorFlow (Keras) model**.  
This project is developed for **academic submission, demonstrations, and practical ML deployment**.

---

## 🚀 Demo
- **Local:** http://127.0.0.1:5000  
- **Live Demo:** Shared temporarily using **ngrok**  
- **Cloud Deployment:** Compatible with **Render**

---

## ✨ Features
- Upload tomato leaf images via web interface  
- Detects **9 tomato leaf conditions**  
- Displays **disease name and confidence score**  
- Simple and clean Flask UI  
- Suitable for academic and demo purposes  

---

## 🧠 Diseases Supported
- Tomato Bacterial Spot  
- Tomato Early Blight  
- Tomato Late Blight  
- Tomato Leaf Mold  
- Tomato Septoria Leaf Spot  
- Tomato Spider Mites  
- Tomato Target Spot  
- Tomato Yellow Leaf Curl Virus  
- Tomato Healthy  

---

## 🛠️ Tech Stack
- **Backend:** Flask (Python)
- **Machine Learning:** TensorFlow 2.10, Keras
- **Image Processing:** OpenCV, Pillow
- **Frontend:** HTML, Jinja Templates
- **Deployment:** ngrok (temporary), Render (permanent)
- **Version Control:** Git & GitHub

---

## 📁 Project Structure
AI-Crop-Disease-Detection/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── tomato_disease_model.h5
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│       └── .gitkeep
│
├── train_model.py
├── predict_single_image.py
└── README.md
