
import random
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page configuration
st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .customer-bubble {
        background: #eef2ff;
        padding: 14px 18px;
        border-radius: 14px;
        margin: 8px 0 18px 0;
    }
    .ai-bubble {
        background: #f0fdf4;
        padding: 14px 18px;
        border-radius: 14px;
        margin: 8px 0 18px 0;
    }
    .metric-card {
        padding: 12px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Model / artifact loading
@st.cache_resource
def load_artifacts():
    model = load_model("customer_support_lstm.keras")

    with open("Customer Support Intent Classifier/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("Customer Support Intent Classifier/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("Customer Support Intent Classifier/responses.pkl", "rb") as f:
        responses = pickle.load(f)

    return model, tokenizer, label_encoder, responses


try:
    model, tokenizer, label_encoder, responses = load_artifacts()
except Exception as e:
    st.error("The AI model files could not be loaded.")
    st.info(
        "Place these four files in the same folder as app.py: "
        "customer_support_lstm.keras, tokenizer.pkl, "
        "label_encoder.pkl, responses.pkl"
    )
    st.exception(e)
    st.stop()


# Configuration

MAX_LENGTH = 20
DEFAULT_THRESHOLD = 0.30


# Prediction
def predict_intent(message):
    sequence = tokenizer.texts_to_sequences([message])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(padded, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    intent = label_encoder.inverse_transform([predicted_index])[0]

    top_indices = np.argsort(probabilities)[-3:][::-1]

    top_predictions = []
    for index in top_indices:
        top_intent = label_encoder.inverse_transform([int(index)])[0]
        top_predictions.append(
            (top_intent, float(probabilities[index]))
        )

    return intent, confidence, top_predictions


# Professional fallback responses
PROFESSIONAL_FALLBACK = (
    "Thank you for contacting our customer support team. "
    "I’m happy to assist you. Could you please provide a few "
    "more details about your request?"
)


def get_response(intent):
    # First use the responses saved from your dataset.
    if intent in responses and responses[intent]:
        return random.choice(responses[intent])

    # Professional fallback if an intent has no saved response.
    return PROFESSIONAL_FALLBACK


def customer_support(message, threshold):
    intent, confidence, top_predictions = predict_intent(message)

    if confidence < threshold:
        return {
            "intent": "unknown",
            "confidence": confidence,
            "response": (
                "Thank you for contacting our customer support team. "
                "I’m sorry, but I wasn’t able to confidently identify "
                "your request. Could you please provide a little more "
                "detail so I can assist you?"
            ),
            "top_predictions": top_predictions,
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "response": get_response(intent),
        "top_predictions": top_predictions,
    }


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Predictions below this confidence are treated as unknown."
    )

    st.divider()

    st.subheader("Model")
    st.write("Architecture: LSTM")
    st.write(f"Intents: {len(label_encoder.classes_)}")
    st.write(f"Sequence length: {MAX_LENGTH}")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.caption(
        "AI Customer Support — Intent classification and "
        "professional response generation."
    )


# Main UI
st.markdown(
    '<div class="main-title">🤖 AI Customer Support</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Ask a question and the LSTM model will '
    'predict the customer intent and provide an appropriate response.</div>',
    unsafe_allow_html=True
)

# Example prompts
st.subheader("Try an example")

examples = [
    "I want to track my order",
    "I forgot my password",
    "My payment failed",
    "Can you tell me more about this product?",
]

cols = st.columns(4)
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.example_message = example

# Input
default_message = st.session_state.pop("example_message", "")

message = st.chat_input("Type your customer message...")

if message is None and default_message:
    message = default_message

# Conversation history
if st.session_state.messages:
    st.subheader("Conversation")

    for item in st.session_state.messages:
        if item["role"] == "customer":
            with st.chat_message("user"):
                st.write(item["content"])
        else:
            with st.chat_message("assistant"):
                st.write(item["content"])

# Process new message
if message and message.strip():
    message = message.strip()

    result = customer_support(message, threshold)

    st.session_state.messages.append({
        "role": "customer",
        "content": message
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["response"]
    })

    st.session_state.last_result = result
    st.rerun()


# ---------------------------------------------------------
# Prediction details
# ---------------------------------------------------------
if st.session_state.last_result:
    result = st.session_state.last_result

    st.divider()
    st.subheader("🔎 Prediction Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Predicted Intent", result["intent"])

    with c2:
        st.metric("Confidence", f"{result['confidence']:.2%}")

    with c3:
        status = "Accepted" if result["intent"] != "unknown" else "Needs clarification"
        st.metric("Status", status)

    st.progress(
        min(max(result["confidence"], 0.0), 1.0),
        text=f"Confidence: {result['confidence']:.2%}"
    )

    with st.expander("View top predictions"):
        for intent, confidence in result["top_predictions"]:
            st.write(f"**{intent}** — {confidence:.2%}")
            st.progress(min(max(confidence, 0.0), 1.0))

    st.caption(
        "The confidence score is the LSTM softmax probability and should "
        "not be interpreted as a guarantee of correctness."
    )
