# MyocardAI: CNN-Based Heart Attack Detection Using ECG Images

MyocardAI is an automated, deep learning-powered medical screening tool designed to analyze and classify ECG images. By leveraging computer vision and Convolutional Neural Networks (CNNs), the system detects cardiac abnormalities and signs of Myocardial Infarction (MI), providing healthcare professionals with a fast, reliable, and consistent second opinion.

---

## 📌 Table of Contents
- [Why MyocardAI?](#-why-myocardai)
- [Project Objectives](#-project-objectives)
- [System Workflow](#%EF%B8%8F-system-workflow)
- [Dataset & Preprocessing Pipeline](#-dataset--preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Training Configuration & Performance](#-training-configuration--performance)
- [Future Scope](#-future-scope)
- [Tech Stack](#-tech-stack)

---

## 💔 Why MyocardAI?
* **Leading Cause of Death:** Heart attacks remain one of the primary causes of mortality worldwide.
* **Time-Critical Survival:** Early detection and rapid intervention significantly improve patient survival rates.
* **Expertise Bottleneck:** Manual interpretation of electrocardiograms (ECGs) requires specialized expertise, is prone to human error, and is highly time-consuming.
* **Accessibility Issues:** Remote and rural areas often face a severe shortage of cardiologists. 

MyocardAI bridges this gap by offering automated, rapid, and highly reliable ECG interpretation.

---

## 🎯 Project Objectives
1. **Automated Detection:** Instantly classify ECG images into distinct health states.
2. **Speed Up Diagnosis:** Significantly reduce the time required to interpret ECG patterns compared to manual review.
3. **Minimize Human Error:** Provide high-precision diagnostic support to prevent missed diagnoses.
4. **Assist Remote Care:** Empower clinical personnel in underserved areas to perform fast cardiac screenings.

---

## ⚙️ System Workflow
The end-to-end pipeline takes a raw ECG image and outputs a classification prediction through the following stages:

[Raw ECG Image]
│
▼
[Image Preprocessing] (Remove grids, noise, standardize shape)
│
▼
[CNN Model Inference] (Automatically extract visual features)
│
▼
[Classification Layer] (Dense / Softmax output)
│
▼
[Final Prediction] ──► [ Normal | Myocardial Infarction (MI) | Abnormal ]
---

## 🖼️ Dataset & Preprocessing Pipeline
Raw clinical ECGs contain noise and grids that can obscure signal waveforms and confuse deep learning models. MyocardAI uses a dedicated preprocessing pipeline to clean raw images:

1. **Grid Removal:** Isolates the active waveform signal by removing the red background ECG grid lines.
2. **Grayscale Conversion:** Simplifies pixel space and reduces computational complexity.
3. **Noise Removal:** Filters out high-frequency artifacts and scan noise.
4. **Resizing:** Standardizes image dimensions to conform to model input parameters.
5. **Normalization:** Normalizes pixel values to improve convergence stability during training.

### Output Classes
The model is trained to classify preprocessed images into three categories:
* **Normal**
* **Myocardial Infarction (MI)**
* **Abnormal** (other cardiac anomalies)

---

## 🧠 Model Architecture
The custom **Convolutional Neural Network (CNN)** automatically maps structural waveform shapes and rhythmic patterns directly from the input image:

* **Input:** Preprocessed ECG Image
* **Feature Extraction:** Multiple Convolutional layers utilizing **ReLU** activation functions paired with **Max-Pooling** operations to capture local visual patterns.
* **Dense Classifier:** Flattened features are fed into a Fully Connected (Dense) network.
* **Output:** **Softmax** activation layer classifying the input into one of the three target classes.

---

## 📈 Training Configuration & Performance

### Training Parameters
* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy
* **Activation (Hidden Layers):** ReLU
* **Activation (Output):** Softmax

### Model Performance
The classifier demonstrated strong overall performance, as shown in the validation test confusion matrix:

| Actual \ Predicted | Normal | MI (Myocardial Infarction) | Abnormal |
| :--- | :---: | :---: | :---: |
| **Normal** | **91%** | 4% | 5% |
| **MI** | 3% | **93%** | 4% |
| **Abnormal** | 6% | 5% | **89%** |

---

## 🚀 Future Scope
To scale MyocardAI into a production-grade clinical application, future updates will focus on:
* **Disease Scope:** Expanding classification classes to detect additional specific heart conditions (e.g., arrhythmias, bundle branch blocks).
* **Mobile Integration:** Developing a lightweight version of the model to run on smartphones for field deployments.
* **Cloud & Real-time Deployment:** Moving inference to cloud environments to allow real-time ECG telemetry analysis.
* **Explainable AI (XAI):** Implementing attention mapping techniques (such as Grad-CAM) to visually highlight which regions of the ECG waveform influenced the model's prediction.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** OpenCV

---

*Presented by Yuvraj Kumar*
