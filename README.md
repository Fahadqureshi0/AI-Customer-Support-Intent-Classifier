# 🤖 AI Customer Support Intent Classifier

<p align="center">
  <strong>An AI-powered customer support system using an LSTM model to classify customer intent and provide professional responses.</strong>
</p>

<p align="center">
  <a href="https://ai-customer-support-intent-classifier-etcm8es7qjalwytm7yx494.streamlit.app/">
     <strong>Live Demo</strong>
  </a>
</p>

---

## 📌 Overview

**AI Customer Support Intent Classifier** is an NLP-based predictive system that takes a customer's message, preprocesses the text, converts it into a numerical sequence, sends it through an **LSTM (Long Short-Term Memory)** model, predicts the customer's intent, and selects an appropriate professional response.

### 🔄 End-to-End Pipeline

```text
Customer Message
       ↓
NLP Preprocessing
       ↓
Tokenizer
       ↓
Embedding
       ↓
LSTM
       ↓
Intent Classification
       ↓
Find Appropriate Answer
       ↓
Professional Customer Response
```

---

## 🚀 Live Demo

### [👉 Open the Interactive Streamlit App](https://ai-customer-support-intent-classifier-etcm8es7qjalwytm7yx494.streamlit.app/)

The interactive application allows users to enter customer-support messages and view:

- 💬 Customer message
- 🧠 Predicted intent
- 📊 Prediction confidence
- 🤖 Professional AI response
- 🔎 Top predicted intents
- ⚙️ Adjustable confidence threshold
- 🗑️ Conversation reset

---

## 🧠 Machine Learning Model

### LSTM — Long Short-Term Memory

The main deep-learning model used in this project is **LSTM (Long Short-Term Memory)**.

The model processes tokenized customer messages as sequences and learns patterns associated with different customer-support intents.

### Model Architecture

```text
Input Text
    ↓
Tokenizer
    ↓
Padded Sequence
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dropout
    ↓
Dense Layer
    ↓
Dropout
    ↓
Softmax Output
    ↓
Predicted Intent
```

The trained model is saved as:

```text
customer_support_lstm.keras
```

---

## 💬 Example Prediction

### Customer Message

```text
I want to track my order
```

### Model Prediction

```text
Predicted Intent: track_order
Confidence: 99.67%
```

### AI Response

```text
Certainly. I’d be happy to help you track your order.
Please provide your order number so I can assist you
with the latest delivery status.
```

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Application and ML development |
| TensorFlow / Keras | Deep learning and LSTM |
| NLP | Text processing and intent classification |
| NumPy | Numerical operations |
| Scikit-learn | Label encoding and ML utilities |
| Streamlit | Interactive web UI |
| Pickle | Saving preprocessing and response objects |

---

## 📁 Project Structure

```text
AI-Customer-Support-Intent-Classifier/
│
├── app.py
├── requirements.txt
├── runtime.txt
│
├── customer_support_lstm.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── responses.pkl
│
└── README.md
```

### Important Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application and prediction logic |
| `customer_support_lstm.keras` | Trained LSTM model |
| `tokenizer.pkl` | Saved tokenizer |
| `label_encoder.pkl` | Converts class indexes into intent names |
| `responses.pkl` | Intent-to-response mapping |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime configuration |

---

## 🔄 How the Prediction System Works

### 1. Customer Message

The customer enters a natural-language request.

```text
Can you tell me more about this product?
```

### 2. Tokenization

The saved tokenizer converts the text into numerical tokens.

```text
Text → Integer Sequence
```

### 3. Padding

The sequence is padded to the same length used during training.

```python
padded = pad_sequences(
    sequence,
    maxlen=50,
    padding="post",
    truncating="post"
)
```

### 4. LSTM Prediction

The padded sequence is passed into the trained LSTM.

```text
LSTM → Probability Distribution
```

### 5. Intent Classification

The highest-probability class becomes the predicted intent.

```text
product_information
```

### 6. Response Selection

The predicted intent is used to select an appropriate professional customer-support response.

---

## 🖥️ Interactive Streamlit Features

The web application includes:

- 💬 Chat-style customer interaction
- 🧠 LSTM intent prediction
- 📊 Confidence score
- 🔎 Top predicted intents
- ⚙️ Adjustable confidence threshold
- 🤖 Professional customer responses
- 🗑️ Clear conversation option
- 📱 Interactive web interface

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Customer-Support-Intent-Classifier.git
cd AI-Customer-Support-Intent-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

```text
streamlit
tensorflow==2.21.0
numpy
scikit-learn
```

Make sure you use a Python version compatible with the selected TensorFlow version.

---

## 💾 Saved AI Components

The application uses four important saved components:

```text
customer_support_lstm.keras
        ↓
      LSTM Model

tokenizer.pkl
        ↓
 Text → Numerical Sequence

label_encoder.pkl
        ↓
 Class Index → Intent

responses.pkl
        ↓
 Intent → Professional Response
```

All four files are required for the deployed prediction system.

---

## 🎯 Project Goals

This project demonstrates an end-to-end AI customer-support workflow:

- Natural-language processing
- Text tokenization
- Sequence padding
- Word embeddings
- LSTM deep learning
- Intent classification
- Confidence-based prediction
- Automated response selection
- Interactive Streamlit deployment

---

## 🔮 Future Improvements

Potential extensions include:

- 🗄️ Real order-database integration
- 📦 Real-time order tracking
- 💳 Payment-status lookup
- 🔄 Automated refund processing
- 🌐 Multilingual support
- 💬 Conversation memory
- 📈 Support analytics dashboard
- 🧠 Transformer-based classification
- 👨‍💼 Human-agent escalation
- 🔐 Authentication and user accounts

---


## 👨‍💻 Project Information

| | |
|---|---|
| **Project** | AI Customer Support Intent Classifier |
| **Model** | LSTM (Long Short-Term Memory) |
| **Application** | Streamlit |
| **Domain** | NLP / Customer Support / Intent Classification |
| **Live Demo** | [Open Application](https://ai-customer-support-intent-classifier-etcm8es7qjalwytm7yx494.streamlit.app/) |

---

## 👨‍💻 Author

**Fahad Qureshi**

---

## 🌐 Connect with Me

[GitHub](https://github.com/Fahadqureshi0)

---

## 📄 License

This project is licensed under the MIT License.
