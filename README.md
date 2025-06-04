# 🧠 TRAVIS - AI-Powered Assistant for Visually Impaired Service Agents

TRAVIS is a **voice-driven, AI-powered banking assistant** designed to empower visually impaired bank agents. It processes spoken queries, classifies them into relevant banking categories using a transformer model, and retrieves customer data when needed—providing an **accessible interface with voice and visual support**.

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
- **Admin Panel**: Allows management of customer accounts (*Create, Update, Delete*).

### ♿ **Accessibility Enhancements**
- **Adjustable Font Sizes**
- **High-Contrast Dark Mode**
- **Voice Response Toggle for Optimal Usability**

---

## 🛠 Technology Stack

| Layer        | Technology                     |
|-------------|--------------------------------|
| Frontend    | React, Web Speech API, TTS     |
| Backend     | Node.js (Express), MongoDB     |
| AI Model    | Python (FastAPI), PyTorch |
| Database    | MongoDB                         |

---

## 📦 Installation & Setup

### 🚀 Clone & Install Dependencies

```bash
git clone https://github.com/AmshudharReddy/TRAVIS
cd TRAVIS
```

#### 🔹 Install Frontend
```bash
cd frontend
npm install
```

#### 🔹 Install Backend
```bash
cd ../backend
npm install
```

#### 🔹 Install AI Model Dependencies
```bash
cd ../services
pip install -r requirements.txt
```

📌 **Python Dependencies** (Ensure the following are installed):
```yaml
fastapi
uvicorn
torch
scikit-learn
nltk
pickle5
```

⚠️ **If punkt tokenizer fails to download**, run:
```python
import nltk
nltk.download('punkt')
```

---

## ▶️ Running the Application

| Service    | Command                  | Runs At              |
|------------|--------------------------|----------------------|
| Frontend   | `npm start`               | `http://localhost:3000` |
| Backend    | `node index.js`           | `http://localhost:5000` |
| AI Model   | `python main.py`          | `http://localhost:5001` |

---

## 🔌 API Endpoints

### **POST /api/classify**
Classifies input queries into relevant banking categories.

#### 📥 Request Example:
```json
{
  "query": "Whhat if i lost my cheque book?"
}
```

#### 📤 Response Example:
```json
{
  "category": "top_up_by_cash_or_cheque"
}
```
```json
{
  "response": "if you lost your cheque deposit please provide the deposit date amount and method used i will review your account and investigate the issue"
}
```

---

## 📸 Screenshots (Optional)
Include visuals to showcase:
- **Agent Dashboard UI**
- **Voice Input & Output Workflow**
- **Dark Mode & Accessibility Features**

---

## 🙋‍♂️ Author
**Amshudhar A.**  
Building accessible, intelligent tools for real-world impact.

## 📜 License
MIT License – See `LICENSE` file for details.

## 📌 Acknowledgements
- PyTorch
- FastAPI
- React
- MongoDB
- Web Speech API
- gTTS
