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
