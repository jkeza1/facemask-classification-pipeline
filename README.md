# Face Mask Classification Pipeline

![Demo GIF](assets/demo.gif) <!-- Add a demo GIF if available -->

An end-to-end machine learning pipeline for classifying face mask usage in images, deployed on the cloud with retraining capabilities and load testing.

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [UI Screenshots](#ui-screenshots)
- [Load Testing Results](#load-testing-results)
- [Notebook Insights](#notebook-insights)
- [Model Files](#model-files)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This project builds, evaluates, and deploys a **Convolutional Neural Network (CNN)** model to classify images into three categories:
1. **Mask worn incorrectly**  
2. **With mask**  
3. **Without mask**  

The solution covers the entire ML lifecycle:  
- Data acquisition → Preprocessing → Model training → API deployment → UI → Retraining → Load testing.

---

## Problem Statement
Manual monitoring of face mask compliance is inefficient and error-prone. This automated system:
- Detects **correct/incorrect/no mask usage** in real-time.  
- Scales via cloud deployment and Docker containers.  
- Adapts to new data with **one-click retraining**.  

---

## Features
| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Model Prediction**   | Upload an image → Get classification (API/UI).                              |
| **Bulk Retraining**    | Upload new images → Trigger model retraining via UI button or API.          |
| **Data Visualizations**| Interactive dashboards for dataset insights (e.g., class distribution).     |
| **Load Tested**        | Simulated 1000+ RPS with Locust; compared Docker vs. no-Docker performance. |

---

## Tech Stack
- **Backend**: Python, FastAPI, TensorFlow/Keras  
- **UI**: Streamlit (or Dash)  
- **Infra**: Docker, Render (AWS/GCP optional)  
- **Testing**: Locust, pytest  
- **Data**: Custom dataset + [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)  

---

## Installation
### Prerequisites
- Python 3.8+
- Docker (for containerization)
- Locust (for load testing)

```bash
# Clone repo
git clone https://github.com/yourusername/facemask-classification-pipeline.git
cd facemask-classification-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
