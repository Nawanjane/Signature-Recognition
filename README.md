# 🖋️ Signature Verification & Attendance Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-17.0.2-blue.svg)](https://reactjs.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Latest-green.svg)](https://www.mongodb.com/)
[![Firebase](https://img.shields.io/badge/Firebase-Latest-orange.svg)](https://firebase.google.com/)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview
An advanced signature verification system integrated with attendance management, powered by deep learning and modern web technologies. The system provides real-time signature authentication and comprehensive attendance tracking capabilities.

### Key Components
1. 🤖 Deep Learning Signature Verification
2. 🎛️ Admin Dashboard (Streamlit)
3. 🌐 User Interface (React + FastAPI)

## ✨ Features

### Signature Verification Model
- 🧠 Advanced CNN Architecture
- 📊 87% Accuracy Rate
- 🔄 Real-time Processing
- 📈 Performance Monitoring

### Admin Panel
- 🔐 Secure Authentication
- 📚 Course Management
- 👥 Student Tracking
- 📊 Analytics Dashboard

### User Interface
- ⚡ Real-time Updates
- 📱 Responsive Design
- 📊 Interactive Charts
- 🖋️ Signature Upload

## 🏗️ Architecture

### System Architecture
```mermaid
graph LR
    A[User Interface] --> B[FastAPI Backend]
    B --> C[MongoDB]
    B --> D[Firebase Auth]
    E[Admin Panel] --> C
    E --> D
    B --> F[ML Model]
```

## 🧠 Model Architecture Details

### CNN Architecture
```ascii
Input (128x128, grayscale)
    ↓
Conv2D (32 filters) + ReLU
    ↓
BatchNorm + MaxPool + Dropout(0.3)
    ↓
Conv2D (64 filters) + ReLU
    ↓
BatchNorm + MaxPool + Dropout(0.3)
    ↓
Dense (128) + ReLU
    ↓
Output (Sigmoid)
```

