# 🤟 SignSpeak: Interactive ASL Learning

A privacy-first, browser-based web application for learning the **static American Sign Language (ASL) alphabet** through real-time hand gesture recognition. Using your webcam, Google MediaPipe, and a custom-trained TensorFlow.js neural network, SignSpeak provides instant feedback as you practice finger-spelling—all while keeping every stage of computation entirely on your device.

Unlike cloud-based vision systems, **no webcam frames or personal data are ever uploaded to a server**. The complete machine learning pipeline executes locally within your browser.

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3 (Responsive UI)
- Vanilla JavaScript (ES6)

### Computer Vision
- [Google MediaPipe Hands](https://github.com/google/mediapipe)
- 21-point 3D Hand Landmark Detection

### Machine Learning
- [TensorFlow.js](https://github.com/tensorflow/tfjs)
- Custom-trained Keras Sequential Neural Network (MLP)

### Additional APIs
- Web Speech API (Text-to-Speech)

---

## 🔄 System Pipeline Architecture

```text
🎥 Webcam Stream
      │
      ▼
🖐️ Google MediaPipe Hands
      │
      ├── Detects 21 3D Hand Landmarks (x, y, z)
      │
      ▼
📐 Anchor Math Preprocessing
      │
      ├── Wrist shifted to (0,0,0)
      ├── Translation removed
      └── Scale normalization
      │
      ▼
📊 Z-Score Normalization
      │
      ├── Uses exported statistics
      └── Matches training pipeline
      │
      ▼
🧠 TensorFlow.js Neural Network
      │
      ├── Dense + Dropout MLP
      └── Predicts ASL Letter
      │
      ▼
💻 Interactive Flashcard Interface
      │
      ├── Displays prediction
      ├── Speaks detected letter
      └── Advances after successful recognition
```

---

## ✨ Core Features

- 🎥 **Real-Time Hand Tracking**
  - Utilizes Google MediaPipe to extract **21 three-dimensional hand landmarks** in real time.

- 🔒 **100% On-Device AI**
  - Powered by TensorFlow.js, all neural network inference runs locally inside the browser.
  - No images, videos, or personal data are uploaded to any server.

- 🎮 **Interactive Flashcard Learning**
  - Practice each ASL letter through a responsive flashcard interface.
  - The application automatically advances once the correct sign is detected.

- 📍 **Position & Distance Independent Recognition**
  - Custom Anchor Math preprocessing removes positional and scaling differences, allowing recognition regardless of where the hand appears in the frame.

- 🔊 **Smart Voice Feedback**
  - Built-in Text-to-Speech announces recognized letters while intelligently avoiding repeated speech triggers.

---

## 🧠 AI Training Pipeline

The neural network powering SignSpeak was trained using a stabilized preprocessing pipeline designed for reliable real-time browser inference.

### 1. Landmark Extraction

Google MediaPipe Hands processes each training image and extracts:

- 21 hand landmarks
- 3 coordinates per landmark (`x`, `y`, `z`)

These landmarks are exported into a structured CSV dataset for training.

---

### 2. Anchor Math

To remove dependence on camera position:

- The wrist landmark (Landmark 0) becomes the origin `(0,0,0)`.
- Every landmark is translated relative to the wrist.
- The hand is scaled by its maximum dimension.

This allows the model to learn the **shape of the hand** instead of its absolute position.

---

### 3. Z-Score Normalization

Each feature is standardized using:

```text
(value - mean) / standard deviation
```

The calculated statistics are exported to:

```text
norm_stats.json
```

The web application applies these same statistics during inference, ensuring identical preprocessing to the training pipeline.

---

### 4. Model Architecture

The classifier is implemented as a lightweight **Keras Sequential Multi-Layer Perceptron (MLP)** consisting of:

- Dense Layers
- Dropout Layers

> **Note:** `BatchNormalization` was intentionally omitted due to known TensorFlow.js export compatibility issues.

---

### 5. TensorFlow.js Export

The trained model is exported directly into TensorFlow.js format:

- `model.json`
- `group1-shard1of1.bin`

This enables fully client-side inference without requiring any backend server.

---

## 📁 Project Structure

```text
asl-on-device-interpreter/
│
├── index.html               # Main application interface
├── style.css                # UI styling
├── script.js                # Real-time inference & flashcard logic
│
├── model.json               # TensorFlow.js model architecture
├── group1-shard1of1.bin     # Trained model weights
├── norm_stats.json          # Normalization statistics
│
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/asl-on-device-interpreter.git
cd asl-on-device-interpreter
```

Replace `yourusername` with your GitHub username.

---

### 2. Verify the Model Files

Ensure the following files are located in the project root:

```text
model.json
group1-shard1of1.bin
norm_stats.json
```

---

### 3. Start a Local Server

Because browsers block local file loading (`file://`) for model assets due to CORS restrictions, the project must be served through a local web server.

If Python is installed:

```bash
python -m http.server 8000
```

---

### 4. Open the Application

Navigate to:

```text
http://localhost:8000
```

Click **Start Camera**, grant webcam permission, and begin practicing the ASL alphabet.

---

## 🔐 Privacy

SignSpeak is designed with privacy as a core principle.

- ✅ No cloud processing
- ✅ No backend server
- ✅ No webcam uploads
- ✅ No image storage
- ✅ No biometric data collection
- ✅ 100% on-device inference

---

## ⚖️ Licenses and Acknowledgements

This project utilizes several open-source resources, frameworks, and datasets. The original source code authored in this repository is released under the MIT License.

Third-party components and datasets are gratefully acknowledged as follows:

- **ASL Training Dataset:** The static neural network was trained using the **[American Sign Language Letters Dataset](https://data.mendeley.com/datasets/jdyksv2jhh/1)** sourced from Mendeley Data. This dataset is legally licensed and utilized under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

- **Computer Vision Framework:** Hand tracking and landmark extraction are powered by **[Google MediaPipe](https://github.com/google/mediapipe)**, which is licensed under the Apache License 2.0.

- **Machine Learning Backend:** On-device neural network inference in the browser is powered by **[TensorFlow.js](https://github.com/tensorflow/tfjs)**, which is licensed under the Apache License 2.0.

---

## 🌟 Future Improvements

- Dynamic ASL word recognition
- Sentence-level interpretation
- Progress tracking
- Difficulty levels and quizzes
- Mobile optimization
- Confidence visualization
- Support for additional sign languages
- Custom practice mode

---

Built with ❤️ to make learning American Sign Language more interactive, accessible, and privacy-friendly.
