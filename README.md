# 🧠 TRAVIS - AI-Powered Assistant for Visually Impaired Service Agents

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org)
[![Node.js](https://img.shields.io/badge/Node.js-16.0+-green.svg)](https://nodejs.org)

TRAVIS is a **voice-driven, AI-powered banking assistant** designed to empower visually impaired bank agents. It processes spoken queries, classifies them into relevant banking categories using a transformer model, and retrieves customer data when needed—providing an **accessible interface with voice and visual support**.

</div>

---

## 🚀 Key Features

### 🎙️ **Voice Interaction**
- **Speech-to-Text**: Uses **Web Speech API** for converting voice input into text.
- **Text-to-Speech (TTS)**: Provides voice responses for seamless communication.
- **Auto-Read Mode**: Toggleable feature to read responses aloud automatically.

### 🧠 **AI-Powered Query Handling**
- **Query Classification**: Uses a transformer-based model for accurate banking category identification.
- **Multi-Mode Response System**:
  - **Transformer Mode**: Generates dynamic responses based on a custom PyTorch model.
  - **Database Mode**: Fetches account-related details for queries requiring authentication.

### 🔁 **Banking Services Covered**
- 💰 **Balance Inquiry**
- 📄 **Account Statement**
- 📌 **KYC Status**
- 🏦 **Loan Approval & Status**

### 👤 **Agent & Admin Dashboard**
- **Agent Profile**: Accessible dashboard for visually impaired bank agents.
- **Admin Panel**: Allows management of customer accounts (*CRUD operations*).

### ♿ **Accessibility Enhancements**
- **Adjustable Font Sizes**
- **High-Contrast Dark Mode**
- **Voice Response Toggle for Optimal Usability**

---

## 🛠 Technology Stack

<div align="center">

| **Layer** | **Technology** |
|-----------|----------------|
| **Frontend** | ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) Web Speech API, TTS |
| **Backend** | ![Node.js](https://img.shields.io/badge/Node.js-43853D?style=flat&logo=node.js&logoColor=white) ![Express.js](https://img.shields.io/badge/Express.js-404D59?style=flat) MongoDB |
| **AI Model** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) |
| **Database** | ![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white) |

</div>

---

## 📦 Installation & Setup

### 🚀 Quick Start

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AmshudharReddy/TRAVIS.git
cd TRAVIS
```

#### 2️⃣ Frontend Setup
```bash
cd frontend
npm install
npm audit fix  # Fix any vulnerabilities
```

#### 3️⃣ Backend Setup
```bash
cd ../backend
npm install
```

#### 4️⃣ AI Services Setup
```bash
cd ../services
pip install -r requirements.txt
```

### 🐍 Python Dependencies
Create a `requirements.txt` file with the following dependencies:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
scikit-learn==1.3.2
nltk==3.8.1
pickle5==0.0.12
transformers==4.35.0
numpy==1.24.3
pandas==2.0.3
```

#### 🔧 NLTK Setup
If you encounter issues with NLTK tokenizers:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## ▶️ Running the Application

### 🖥️ Development Mode
Open **three separate terminals** and run:

<div align="center">

| **Service** | **Directory** | **Command** | **URL** |
| ----------- | ------------- | ----------- | ------- |
| 🎨 **Frontend** | `./frontend` | `npm start` | `http://localhost:3000` |
| 🔧 **Backend** | `./backend` | `node index.js` | `http://localhost:5000` |
| 🤖 **AI Services** | `./ai_services` | `python main.py` | `http://localhost:5001` |

</div>

---

## 🔌 API Endpoints

### 📥 **Request Example:**
```json
{
  "query": "What if I lost my cheque book?"
}
```

### 📤 **Response Examples:**

#### 1️⃣ **Encoder-only Transformer model:**
*Classifies input queries into relevant banking categories.*
```json
{
  "category": "top_up_by_cash_or_cheque"
}
```

#### 2️⃣ **Encoder-Decoder Transformer (Seq2Seq):**
*Generates a response for the given input query.*
```json
{
  "response": "if you lost your cheque deposit please provide the deposit date amount and method used i will review your account and investigate the issue"
}
```

### 🌐 **Translation Example:**

#### 3️⃣ **Encoder-Decoder Transformer (Seq2Seq):**
*Translates the generated response into the local language (Telugu).*
```json
{
  "translation": "మీరు మీ చెక్కును జమ చేయకపోతే దయచేసి డిపాజిట్ తేదీ మొత్తం మరియు ఉపయోగించిన పద్ధతిని అందించండి నేను మీ ఖాతాను సమీక్షిస్తాను మరియు సమస్యను పరిశీలిస్తాను"
}
```

---

## 🏗️ Architecture

```mermaid
graph TB
    A[User Voice Input] --> B[Web Speech API]
    B --> C[React Frontend]
    C --> D[Node.js Backend]
    
    D --> |AI Mode| E[FastAPI AI Service]
    D --> |Database Mode| J[MongoDB Database]
    
    E --> F[PyTorch Models]
    F --> G[AI Response Generation]
    
    J --> K[Customer Data Retrieval]
    K --> L[Database Response]
    
    G --> M[Text-to-Speech]
    L --> M[Text-to-Speech]
    M --> N[Voice Output]
    
    style A fill:#4fc3f7,stroke:#0277bd,stroke-width:3px,color:#000
    style B fill:#81c784,stroke:#388e3c,stroke-width:3px,color:#000
    style C fill:#64b5f6,stroke:#1976d2,stroke-width:3px,color:#000
    style D fill:#ffb74d,stroke:#f57c00,stroke-width:3px,color:#000
    style E fill:#ba68c8,stroke:#7b1fa2,stroke-width:3px,color:#000
    style F fill:#f06292,stroke:#c2185b,stroke-width:3px,color:#000
    style G fill:#ff8a65,stroke:#d84315,stroke-width:3px,color:#000
    style J fill:#4db6ac,stroke:#00695c,stroke-width:3px,color:#000
    style K fill:#26a69a,stroke:#004d40,stroke-width:3px,color:#000
    style L fill:#66bb6a,stroke:#2e7d32,stroke-width:3px,color:#000
    style M fill:#ffd54f,stroke:#f9a825,stroke-width:3px,color:#000
    style N fill:#a5d6a7,stroke:#388e3c,stroke-width:3px,color:#000
```
---

## 📸 Screenshots

<div align="center">

### 🖥️ **Agent Dashboard UI**
![Agent Dashboard](https://github.com/user-attachments/assets/e4ffd5fb-fc48-4ca4-ae8b-820fd50d6252)

### 💬 **Response Display UI**
![Response Display](https://github.com/user-attachments/assets/c3a0772b-0034-49a3-99ba-3880fd1406dd)

### 🔄 **Input-Output Workflow**
![Input-Output Workflow](https://github.com/user-attachments/assets/d40d6fb9-7eb0-44ac-abc6-d1dda808b3c9)

### 🌙 **Dark Mode & Accessibility Features**
![Dark Mode Features](https://github.com/user-attachments/assets/576391d1-5e97-4363-811a-0f6aced389ca)

</div>

---
## 🛣️ Roadmap

### 🎯 Version 2.0 Goals

- [ ] **Multi-language Voice Support** - Additional regional languages
- [ ] **Advanced Analytics Dashboard** - Usage insights and reporting
- [ ] **Mobile Application** - Native iOS and Android apps
- [ ] **Biometric Authentication** - Voice pattern recognition
- [ ] **Real-time Notifications** - Account alerts and updates
- [ ] **Enhanced AI model accuracy**
- [ ] **Improved accessibility features**
- [ ] **Advanced security measures**
- [ ] **Performance optimizations**

---

## 🚨 Troubleshooting

### Common Issues

**Voice recognition not working?**
- Ensure microphone permissions are granted
- Check browser compatibility (Chrome/Edge recommended)
- Verify microphone hardware functionality

**AI service connection failed?**
- Confirm Python dependencies are installed
- Check if port 5001 is available
- Verify model files are present

**Database connection issues?**
- Ensure MongoDB is running
- Check connection string in `.env`
- Verify database permissions

---

## 📜 License
**MIT License** – See `LICENSE` file for details.

## 📌 Acknowledgements

We extend our gratitude to the following technologies and communities:

<div align="center">

| Technology | Purpose | Links |
|------------|---------|-------|
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep Learning Framework | [pytorch.org](https://pytorch.org) |
| ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) | Modern Python API Framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) | Frontend Library | [reactjs.org](https://reactjs.org) |
| ![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=flat&logo=mongodb&logoColor=white) | Document Database | [mongodb.com](https://mongodb.com) |
| **Web Speech API** | Browser Speech Recognition | [MDN Docs](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) |
| **gTTS** | Google Text-to-Speech | [PyPI](https://pypi.org/project/gTTS/) |

</div>

## 🤝 Contributing

We appreciate contributions via pull requests. For major changes, please open an issue first so we can have a quick discussion about your plans.

---

## 🙋‍♂️ Author

<div align="center">

**Amshudhar A. & Team**  
*Building accessible, intelligent tools for real-world impact.*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AmshudharReddy)

</div>

---

<div align="center">

### 🌟 If TRAVIS helped you, please consider giving it a star!

[![GitHub stars](https://img.shields.io/github/stars/AmshudharReddy/TRAVIS.svg?style=social&label=Star)](https://github.com/AmshudharReddy/TRAVIS)

**Made with ❤️ for accessibility and inclusion**

</div>
