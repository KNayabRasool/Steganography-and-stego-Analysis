import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# LSB Steganography Functions

def encode_message(img: Image.Image, message: str) -> Image.Image:
    img = img.convert("RGB")
    img_array = np.array(img)
    flat_pixels = img_array.flatten()

    msg_bytes = message.encode('utf-8')
    msg_len = len(msg_bytes)
    header = format(msg_len, '032b')
    payload = ''.join(format(b, '08b') for b in msg_bytes)
    binary_message = header + payload

    if len(binary_message) > len(flat_pixels):
        raise ValueError("Message too long for this image!")

    for i, bit in enumerate(binary_message):
        pv = int(flat_pixels[i])
        flat_pixels[i] = np.uint8((pv & ~1) | int(bit))

    encoded_img = Image.fromarray(flat_pixels.reshape(img_array.shape))
    return encoded_img


def decode_message(img: Image.Image) -> str:
    img = img.convert("RGB")
    img_array = np.array(img)
    flat_pixels = img_array.flatten()

    bits = [str(int(pix) & 1) for pix in flat_pixels]
    msg_len = int(''.join(bits[:32]), 2)
    total_bits = 32 + msg_len * 8
    msg_bits = bits[32:total_bits]

    bytes_list = [int(''.join(msg_bits[i:i + 8]), 2) for i in range(0, len(msg_bits), 8)]
    return bytes(bytes_list).decode('utf-8')


# Detection Feature Extractor


def extract_features(img: Image.Image):
    img = img.convert("L")
    arr = np.array(img).astype(float)
    mean = np.mean(arr)
    std = np.std(arr)
    edge_strength = np.mean(np.abs(np.gradient(arr)))
    return [mean, std, edge_strength]


# Train Detection Model (Run Once)


def train_detection_model(data_dir="dataset", save_model=True):
    X, y = [], []
    for label, folder in enumerate(["clean", "stego"]):
        folder_path = os.path.join(data_dir, folder)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(folder_path, f))
                feat = extract_features(img)
                X.append(feat)
                y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    st.write(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

    if save_model:
        with open("stego_model.pkl", "wb") as f:
            pickle.dump(model, f)
        st.write("Model saved as `stego_model.pkl`")
    return model


def load_detection_model():
    if os.path.exists("stego_model.pkl"):
        with open("stego_model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return None


# Streamlit UI

st.set_page_config(page_title="Steganography & Detection Suite", page_icon="🧠", layout="centered")
st.title("🧠 Unified Image Steganography & Detection App")

menu = st.sidebar.selectbox("Choose an Option", ["Hide Message", "Extract Message", "Detect Stego Image", "Train Detector"])

# HIDE MESSAGE

if menu == "Hide Message":
    st.header("🔒 Hide a Secret Message")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    secret_text = st.text_area("Enter secret message:")

    if uploaded_file and secret_text:
        img = Image.open(uploaded_file)
        st.image(img, caption="Original Image", use_container_width=True)

        if st.button("Encode Message"):
            try:
                encoded_img = encode_message(img, secret_text)
                st.success("Message hidden successfully!")

                buf = io.BytesIO()
                encoded_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button("📥 Download Encoded Image", data=byte_im, file_name="encoded.png", mime="image/png")
            except Exception as e:
                st.error(f"Error: {e}")



# EXTRACT MESSAGE


elif menu == "Extract Message":
    st.header("🔓 Extract Hidden Message")
    uploaded = st.file_uploader("Upload Encoded Image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Encoded Image", use_container_width=True)

        if st.button("Decode Message"):
            try:
                message = decode_message(img)
                st.success("Message extracted successfully!")
                st.text_area("Hidden Message:", value=message, height=150)
            except Exception as e:
                st.error(f"Error: {e}")



# DETECT STEGO IMAGE


elif menu == "Detect Stego Image":
    st.header("🕵️ Detect Hidden Data in Image")
    uploaded = st.file_uploader("Upload Image to Analyze", type=["png", "jpg", "jpeg"])

    model = load_detection_model()
    if not model:
        st.warning("⚠️ No trained model found. Please train one first using 'Train Detector' tab.")
    elif uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            features = extract_features(img)
            pred = model.predict([features])[0]
            if pred == 1:
                st.error("⚠️ Steganographic content detected (Image is likely encoded).")
            else:
                st.success("✅ No hidden data detected (Clean Image).")

# TRAIN DETECTOR


else:
    st.header("🧩 Train Steganography Detection Model")
    st.write("""
    Prepare a dataset folder with this structure:
    ```
    dataset/
      ├── clean/   -> Original images (no hidden data)
      └── stego/   -> Images with hidden messages
    ```
    Then click below to train.
    """)
    if st.button("Train Model"):
        train_detection_model()
