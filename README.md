# Chest X-ray Pneumonia Detection

AI-powered Chest X-ray Pneumonia Detection web application using EfficientNetB0 transfer learning and Flask.

## 🚀 Features

- **Deep Learning Model:** EfficientNetB0 fine-tuned on chest X-ray images.
- **Binary Classification:** Detects Normal vs. Pneumonia with confidence score and probability distributions.
- **Modern Interactive UI:** Drag-and-drop file upload, real-time feedback, and diagnostic metrics.
- **Production Ready:** Configured for local running and cloud deployment on Render via Gunicorn.

## 📁 Repository Structure

```
├── model/
│   └── EfficientNetB0_Pneumonia_Final.keras
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
└── .gitignore
```

## 💻 Local Setup & Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. Open **http://localhost:5000** in your browser.

## ☁️ Deployment on Render

1. Create a new **Web Service** on [Render](https://render.com/).
2. Connect your GitHub repository: `https://github.com/abdl-hafeez07/PneumoniaDetection.git`.
3. Set configuration:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Health Check Path:** `/health`
