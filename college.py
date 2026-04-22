import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 🔗 BLOCKCHAIN
# -------------------------------
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.create_hash()

    def create_hash(self):
        return hashlib.sha256(
            f"{self.index}{self.timestamp}{self.data}{self.previous_hash}".encode()
        ).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, str(datetime.datetime.now()), "Genesis Block", "0")

    def add_block(self, data):
        prev = self.chain[-1]
        new_block = Block(len(self.chain), str(datetime.datetime.now()), data, prev.hash)
        self.chain.append(new_block)


# -------------------------------
# INIT
# -------------------------------
if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()

st.set_page_config(page_title="☕ Coffee AI Pro", layout="wide")
st.title("☕ Smart Coffee AI + Blockchain System")

# -------------------------------
# 📊 DATA
# -------------------------------
np.random.seed(42)

df = pd.DataFrame({
    "aroma": np.random.uniform(6, 10, 100),
    "flavor": np.random.uniform(6, 10, 100),
    "acidity": np.random.uniform(6, 10, 100),
    "body": np.random.uniform(6, 10, 100),
    "balance": np.random.uniform(6, 10, 100),
})

df["quality"] = (
    df["aroma"] * 0.25 +
    df["flavor"] * 0.30 +
    df["acidity"] * 0.15 +
    df["body"] * 0.15 +
    df["balance"] * 0.15
) * 10

features = ["aroma", "flavor", "acidity", "body", "balance"]

# -------------------------------
# 📊 TRAIN MODEL
# -------------------------------
model = RandomForestRegressor()
model.fit(df[features], df["quality"])

# -------------------------------
# 📊 DATA TABLE (FIXED MISSING PART)
# -------------------------------
st.subheader("📊 Coffee Dataset")
st.dataframe(df)

# -------------------------------
# 🎛 INPUT
# -------------------------------
st.sidebar.header("Coffee Inputs")

aroma = st.sidebar.slider("Aroma", 1.0, 10.0, 7.0)
flavor = st.sidebar.slider("Flavor", 1.0, 10.0, 7.0)
acidity = st.sidebar.slider("Acidity", 1.0, 10.0, 7.0)
body = st.sidebar.slider("Body", 1.0, 10.0, 7.0)
balance = st.sidebar.slider("Balance", 1.0, 10.0, 7.0)

# -------------------------------
# 🤖 PREDICTION
# -------------------------------
if st.sidebar.button("Predict Quality"):
    input_data = np.array([[aroma, flavor, acidity, body, balance]])
    prediction = model.predict(input_data)[0]

    st.success(f"☕ Predicted Quality: {prediction:.2f}")

    record = {
        "aroma": aroma,
        "flavor": flavor,
        "acidity": acidity,
        "body": body,
        "balance": balance,
        "quality": float(prediction)
    }

    st.session_state.blockchain.add_block(record)

# -------------------------------
# 📊 GRAPH (WORKING)
# -------------------------------
st.subheader("📊 Feature Importance Graph")

importance = model.feature_importances_

fig = px.bar(
    x=features,
    y=importance,
    text=np.round(importance, 3),
    title="Feature Importance in Coffee Quality Prediction",
    color=importance
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 🔗 BLOCKCHAIN SECTION
# -------------------------------
st.subheader("🔗 Blockchain Explorer")

chain = st.session_state.blockchain.chain

block_labels = [f"Block {b.index}" for b in chain]
selected = st.selectbox("Select Block", block_labels)

col1, col2 = st.columns(2)

with col1:
    if st.button("Delete Selected Block"):
        idx = int(selected.split(" ")[1])

        if idx == 0:
            st.warning("Cannot delete Genesis Block")
        else:
            st.session_state.blockchain.chain = [
                b for b in chain if b.index != idx
            ]

            # reindex
            for i, b in enumerate(st.session_state.blockchain.chain):
                b.index = i

            st.success(f"Block {idx} deleted")

with col2:
    if st.button("Show Block Details"):
        idx = int(selected.split(" ")[1])
        block = st.session_state.blockchain.chain[idx]

        st.json(block.data)
        st.text(block.hash)

# -------------------------------
# FULL CHAIN VIEW
# -------------------------------
st.write("### Full Blockchain History")

for block in st.session_state.blockchain.chain:
    with st.expander(f"Block {block.index}"):
        st.json(block.data)
        st.text(block.hash)