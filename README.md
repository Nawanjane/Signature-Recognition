# Signature Verification and Attendance Management System

## Project Overview
This project combines deep learning-based signature verification with a comprehensive attendance management system. It consists of three main components:
1. Signature Verification Model
2. Admin Panel (Streamlit)
3. User Interface (React + FastAPI)

## 1. Signature Verification Model

### Model Architecture
- Convolutional Neural Network (CNN) with multiple layers
- Input: Grayscale signature images (128x128 pixels)
- Output: Binary classification (genuine/forged)

### Key Features
- Batch Normalization for stable training
- Dropout layers for regularization
- L2 regularization in dense layers
- Custom learning rate scheduler
- F1-score monitoring during training

### Performance Metrics
Based on the training results in <mcfile name="training_history.csv" path="/Users/leopard/Desktop/Cource_Work_2/signature/outputs/CURRUNT/reports/training_history.csv"></mcfile>:
- Accuracy: 87%
- Precision: 88%
- Recall: 87%

### Model Visualization
The following visualizations are available in the outputs directory:
- Training history plots (`/outputs/figures/training_history.png`)
- Confusion matrix (`/outputs/figures/confusion_matrix.png`)
- Sample predictions (`/outputs/figures/example_predictions.png`)

## 2. Admin Panel (Streamlit)

### Features
- User authentication using Firebase
- Course management
- Student attendance tracking
- Signature verification monitoring
- Analytics dashboard

### Implementation
The admin panel is implemented using Streamlit and is located in <mcfolder name="student-attendance-management" path="/Users/leopard/Desktop/Cource_Work_2/signature/APP/student-attendance-management"></mcfolder>:
- `app.py`: Main application file
- Firebase integration for authentication
- MongoDB integration for data storage

## 3. User Interface

### Backend (FastAPI)
Located in <mcfolder name="API" path="/Users/leopard/Desktop/Cource_Work_2/signature/APP/API"></mcfolder>:
- RESTful API endpoints
- WebSocket support for real-time updates
- Authentication middleware
- Database integration

### Frontend (React)
Located in <mcfolder name="attendance-ui" path="/Users/leopard/Desktop/Cource_Work_2/signature/APP/API/attendance-ui"></mcfolder>:
- Material-UI components
- Real-time attendance tracking
- Interactive dashboard with charts
- Signature upload and verification

## Database Architecture

### Firebase
- User authentication
- Role management
- Security rules

### MongoDB Collections
- attendance_records
- courses
- signature_validation

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB
- Firebase account

