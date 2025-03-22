import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from pymongo import MongoClient
import cv2
import pandas as pd
import numpy as np
import pytesseract
import os
import re
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io
from datetime import datetime
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    try:
        # Use JSON file directly
        cred_path = 'your json'
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            st.error(f"Firebase credentials file not found at {cred_path}")
            st.stop()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.stop()

# Connect to MongoDB using environment variable
client = MongoClient(os.getenv('MONGODB_URI'))
db = client['attendance_db']
attendance_collection = db['attendance_records']
courses_collection = db['courses']
signature_collection = db['signature_validation']

# Set page configuration
st.set_page_config(
    page_title="Student Attendance Management",
    page_icon="📋",
    layout="wide"
)

# Create necessary directories
os.makedirs('signatures', exist_ok=True)
os.makedirs('debug', exist_ok=True)

# Function definitions
def ocr_extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()

def clean_text(text):
    text = re.sub(r'[^\w\s.-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def save_signature_image(id_text, signature, time_of_day, course_name=None, attendance_date=None):
    # Create base signatures directory
    os.makedirs('signatures', exist_ok=True)
    
    if course_name and attendance_date:
        # Create nested directory structure: signatures/course_name/date/
        folder_path = os.path.join('signatures', course_name, str(attendance_date))
        os.makedirs(folder_path, exist_ok=True)
        
        if has_sufficient_signature_content(signature):
            filename = os.path.join(folder_path, f"{id_text}_{time_of_day}.png")
            cv2.imwrite(filename, signature, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return True
    else:
        # Fallback to original behavior if course_name or date not provided
        if has_sufficient_signature_content(signature):
            filename = f"signatures/{id_text}_{time_of_day}.png"
            cv2.imwrite(filename, signature, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            return True
    
    st.warning(f"Skipping signature for ID {id_text} ({time_of_day}) - insufficient content")
    return False

def has_sufficient_signature_content(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    signature_pixels = cv2.countNonZero(thresh)
    total_pixels = gray.shape[0] * gray.shape[1]
    signature_percentage = (signature_pixels / total_pixels) * 100
    return signature_percentage >= 20

def extract_info(image_path, course_name, attendance_date):
    image = cv2.imread(image_path)
    original_image = image.copy()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//20, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    dilated_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_positions = [cv2.boundingRect(contour)[1] for contour in contours if cv2.boundingRect(contour)[2] > width * 0.7]
    line_positions.sort()
    debug_image = original_image.copy()
    for i in range(len(line_positions) - 1):
        y_start, y_end = line_positions[i], line_positions[i+1]
        overlay = debug_image.copy()
        cv2.rectangle(overlay, (0, y_start), (width, y_end), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, debug_image, 0.7, 0, debug_image)
        cv2.line(debug_image, (0, y_start), (width, y_start), (0, 0, 255), 2)
    if line_positions:
        cv2.line(debug_image, (0, line_positions[-1]), (width, line_positions[-1]), (0, 0, 255), 2)
    st.image(debug_image, caption="Detected Rows", use_column_width=True)
    line_positions_str = st.text_input("Row positions (comma-separated y-coordinates):", value=",".join(map(str, line_positions)))
    try:
        line_positions = [int(pos.strip()) for pos in line_positions_str.split(",") if pos.strip()]
        line_positions.sort()
    except ValueError:
        st.error("Invalid input. Please enter comma-separated numbers.")
    id_col_start, id_col_end = 0, int(width * 0.25)
    name_col_start, name_col_end = id_col_end, int(width * 0.6)
    morning_col_start, morning_col_end = name_col_end, int(width * 0.8)
    evening_col_start, evening_col_end = morning_col_end, width
    col1, col2 = st.columns(2)
    with col1:
        id_col_end = st.slider("ID Column End", min_value=int(width * 0.1), max_value=int(width * 0.4), value=id_col_end, step=5)
    with col2:
        name_col_end = st.slider("Name Column End", min_value=id_col_end + 10, max_value=int(width * 0.7), value=name_col_end, step=5)
    morning_col_end = st.slider("Morning Column End", min_value=name_col_end + 10, max_value=int(width * 0.9), value=morning_col_end, step=5)
    name_col_start, morning_col_start, evening_col_start = id_col_end, name_col_end, morning_col_end
    column_preview = original_image.copy()
    cv2.line(column_preview, (id_col_end, 0), (id_col_end, height), (0, 0, 255), 2)
    cv2.line(column_preview, (name_col_end, 0), (name_col_end, height), (0, 0, 255), 2)
    cv2.line(column_preview, (morning_col_end, 0), (morning_col_end, height), (0, 0, 255), 2)
    cv2.putText(column_preview, "ID", (id_col_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(column_preview, "Name", (name_col_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(column_preview, "Morning", (morning_col_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(column_preview, "Evening", (evening_col_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    st.image(column_preview, caption="Column Boundaries", use_column_width=True)
    data = []
    if st.button("Process Attendance Sheet"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(len(line_positions) - 1):
            progress = (i + 1) / (len(line_positions) - 1)
            progress_bar.progress(progress)
            status_text.text(f"Processing row {i+1} of {len(line_positions)-1}...")
            row_y_start, row_y_end = line_positions[i], line_positions[i+1]
            row_height = row_y_end - row_y_start
            if row_height < 5:
                st.warning(f"Skipping row {i} - too small (height: {row_height}px)")
                continue
            id_roi = original_image[row_y_start:row_y_end, id_col_start:id_col_end]
            name_roi = original_image[row_y_start:row_y_end, name_col_start:name_col_end]
            morning_sig_roi = original_image[row_y_start:row_y_end, morning_col_start:morning_col_end]
            evening_sig_roi = original_image[row_y_start:row_y_end, evening_col_start:evening_col_end]
            if id_roi.size == 0 or name_roi.size == 0 or morning_sig_roi.size == 0 or evening_sig_roi.size == 0:
                st.warning(f"Skipping row {i} - one or more ROIs are empty")
                continue
            debug_row_dir = f'debug/row_{i}'
            os.makedirs(debug_row_dir, exist_ok=True)
            id_text = ocr_extract_text(id_roi)
            name_text = ocr_extract_text(name_roi)
            id_text = clean_text(id_text)
            name_text = clean_text(name_text)
            id_match = re.search(r'(\d{3})$', id_text)
            if id_match:
                id_number = id_match.group(1)
                formatted_id = f"KIC-HNDDS-241-F-{id_number}"
            else:
                formatted_id = f"unknown_row_{i}"
            morning_has_signature = has_sufficient_signature_content(morning_sig_roi)
            evening_has_signature = has_sufficient_signature_content(evening_sig_roi)
            if id_text or name_text or morning_has_signature or evening_has_signature:
                morning_status = "Present" if morning_has_signature else "Absent"
                evening_status = "Present" if evening_has_signature else "Absent"
                if morning_has_signature:
                    save_signature_image(formatted_id, morning_sig_roi, "morning", course_name, attendance_date)
                if evening_has_signature:
                    save_signature_image(formatted_id, evening_sig_roi, "evening", course_name, attendance_date)
                data.append((formatted_id, name_text, morning_status, evening_status))
                st.info(f"Extracted row {i}: ID={formatted_id}, Name={name_text}, Morning={morning_status}, Evening={evening_status}")
            else:
                st.warning(f"Row {i} appears to be empty or unreadable")
        progress_bar.empty()
        status_text.empty()
        if data:
            df = pd.DataFrame(data, columns=['ID', 'Name', 'Morning', 'Evening'])
            st.success("Extraction completed successfully!")
            st.write("### Extracted Attendance Data")
            st.dataframe(df)
            attendance_data = {
                'course_name': course_name,
                'date': attendance_date,
                'attendance': df.to_dict('records')
            }
            attendance_collection.insert_one(attendance_data)
            st.success(f"Data saved to MongoDB for {course_name} on {attendance_date}!")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{course_name}_{attendance_date}.csv",
                mime="text/csv"
            )
            return df
        else:
            st.error("No data was extracted. Please check the image and try again.")
            return None
    return None

def validate_signature(image_path, model, threshold=0.3):
    try:
        img = cv2.imread(image_path)
        if img is None:
            st.error(f"Failed to load image: {image_path}")
            return "forge", 0.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)[0][0]
        prediction = "genuine" if pred_prob > threshold else "forge"
        return prediction, float(pred_prob)
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        return "forge", 0.0

def validate_signatures(attendance_df, model_path, course_name, attendance_date):
    if not os.path.exists(model_path):
        default_model_path = '/Users/leopard/Desktop/Cource_Work_2/signature/APP/student-attendance-management/best_signature_model.h5'
        if os.path.exists(default_model_path):
            model_path = default_model_path
            st.info(f"Using default model at {model_path}")
        else:
            st.error(f"Model file not found: {model_path}")
            st.error("Please upload a model file or place 'best_signature_model.h5' in the application directory.")
            return attendance_df
    with st.spinner("Loading signature verification model..."):
        try:
            model = load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return attendance_df
    if 'Morning_Prediction' not in attendance_df.columns:
        attendance_df['Morning_Prediction'] = ""
        attendance_df['Morning_Confidence'] = 0.0
    if 'Evening_Prediction' not in attendance_df.columns:
        attendance_df['Evening_Prediction'] = ""
        attendance_df['Evening_Confidence'] = 0.0
    progress_bar = st.progress(0)
    status_text = st.empty()
    for index, row in attendance_df.iterrows():
        progress = (index + 1) / len(attendance_df)
        progress_bar.progress(progress)
        status_text.text(f"Validating signatures for student {index+1} of {len(attendance_df)}...")
        student_id = row['ID']
        
        # Update signature paths to include course and date folders
        course_date_path = os.path.join('signatures', course_name, str(attendance_date))
        
        # Initialize both flags at the start
        morning_found = False
        evening_found = False
        
        # Morning signature validation
        morning_sig_paths = [
            os.path.join(course_date_path, f"{student_id}_morning.png"),
            os.path.join(course_date_path, f"KIC-HNDDS-241-F-{student_id}_morning.png"),
            os.path.join(course_date_path, f"KIC-HNDDS-241-F-{student_id.split('-')[-1]}_morning.png")
        ]
        
        for path in morning_sig_paths:
            if os.path.exists(path):
                prediction, confidence = validate_signature(path, model)
                attendance_df.at[index, 'Morning_Prediction'] = prediction
                attendance_df.at[index, 'Morning_Confidence'] = confidence
                st.info(f"Student {student_id} morning signature: {prediction} (Confidence: {confidence:.4f})")
                morning_found = True
                break
        if not morning_found:
            # Check in course-specific directory
            if os.path.exists(course_date_path):
                for filename in os.listdir(course_date_path):
                    if student_id in filename and "morning" in filename:
                        path = os.path.join(course_date_path, filename)
                        prediction, confidence = validate_signature(path, model)
                        attendance_df.at[index, 'Morning_Prediction'] = prediction
                        attendance_df.at[index, 'Morning_Confidence'] = confidence
                        st.info(f"Student {student_id} morning signature found in course folder: {prediction} (Confidence: {confidence:.4f})")
                        morning_found = True
                        break
            if not morning_found:
                st.warning(f"Morning signature not found for student {student_id}")
        
        if not evening_found:
            # Check in course-specific directory
            if os.path.exists(course_date_path):
                for filename in os.listdir(course_date_path):
                    if student_id in filename and "evening" in filename:
                        path = os.path.join(course_date_path, filename)
                        prediction, confidence = validate_signature(path, model)
                        attendance_df.at[index, 'Evening_Prediction'] = prediction
                        attendance_df.at[index, 'Evening_Confidence'] = confidence
                        st.info(f"Student {student_id} evening signature found in course folder: {prediction} (Confidence: {confidence:.4f})")
                        evening_found = True
                        break
            if not evening_found:
                st.warning(f"Evening signature not found for student {student_id}")
    progress_bar.empty()
    status_text.empty()
    attendance_df['Morning'] = attendance_df.apply(
        lambda row: 'Present' if row['Morning_Prediction'] == 'genuine' else 'Absent', axis=1
    )
    attendance_df['Evening'] = attendance_df.apply(
        lambda row: 'Present' if row['Evening_Prediction'] == 'genuine' else 'Absent', axis=1
    )
    # Save validation results to MongoDB
    validation_data = {
        'course_name': course_name,
        'date': attendance_date,
        'validation_results': attendance_df.to_dict('records')
    }
    signature_collection.insert_one(validation_data)
    st.success("Validation results saved to MongoDB!")
    return attendance_df

def display_signature_gallery(attendance_df, course_name=None, attendance_date=None):
    st.write("### Signature Gallery")
    morning_tab, evening_tab = st.tabs(["Morning Signatures", "Evening Signatures"])
    
    # Get the course-specific signature path
    course_date_path = os.path.join('signatures', course_name, str(attendance_date)) if course_name and attendance_date else 'signatures'
    
    with morning_tab:
        cols = st.columns(3)
        col_idx = 0
        for index, row in attendance_df.iterrows():
            student_id = row['ID']
            morning_sig_path = None
            
            # First check in course-specific directory
            if os.path.exists(course_date_path):
                for filename in os.listdir(course_date_path):
                    if student_id in filename and "morning" in filename:
                        morning_sig_path = os.path.join(course_date_path, filename)
                        break
            
            # Fallback to root signatures directory if not found
            if not morning_sig_path:
                for filename in os.listdir('signatures'):
                    if student_id in filename and "morning" in filename:
                        morning_sig_path = os.path.join('signatures', filename)
                        break
                        
            if morning_sig_path and os.path.exists(morning_sig_path):
                with cols[col_idx]:
                    img = Image.open(morning_sig_path)
                    st.image(img, caption=f"{row['Name']} ({student_id})")
                    if 'Morning_Prediction' in row and row['Morning_Prediction']:
                        prediction = row['Morning_Prediction']
                        confidence = row['Morning_Confidence']
                        color = "green" if prediction == "genuine" else "red"
                        st.markdown(f"<p style='color:{color};'>Prediction: {prediction} ({confidence:.2f})</p>", unsafe_allow_html=True)
                    col_idx = (col_idx + 1) % 3

    with evening_tab:
        cols = st.columns(3)
        col_idx = 0
        for index, row in attendance_df.iterrows():
            student_id = row['ID']
            evening_sig_path = None
            
            # First check in course-specific directory
            if os.path.exists(course_date_path):
                for filename in os.listdir(course_date_path):
                    if student_id in filename and "evening" in filename:
                        evening_sig_path = os.path.join(course_date_path, filename)
                        break
            
            # Fallback to root signatures directory if not found
            if not evening_sig_path:
                for filename in os.listdir('signatures'):
                    if student_id in filename and "evening" in filename:
                        evening_sig_path = os.path.join('signatures', filename)
                        break
                        
            if evening_sig_path and os.path.exists(evening_sig_path):
                with cols[col_idx]:
                    img = Image.open(evening_sig_path)
                    st.image(img, caption=f"{row['Name']} ({student_id})")
                    if 'Evening_Prediction' in row and row['Evening_Prediction']:
                        prediction = row['Evening_Prediction']
                        confidence = row['Evening_Confidence']
                        color = "green" if prediction == "genuine" else "red"
                        st.markdown(f"<p style='color:{color};'>Prediction: {prediction} ({confidence:.2f})</p>", unsafe_allow_html=True)
                    col_idx = (col_idx + 1) % 3

def admin_login():
    st.sidebar.title("Admin Login")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login"):
        try:
            user = auth.get_user_by_email(email)
            st.sidebar.success(f"Welcome {user.email}")
            return True
        except Exception as e:
            st.sidebar.error("Invalid credentials")
            return False

def admin_register():
    st.sidebar.title("Admin Registration")
    email = st.sidebar.text_input("Email", key="register_email")
    password = st.sidebar.text_input("Password", type="password", key="register_password")
    if st.sidebar.button("Register"):
        try:
            user = auth.create_user(email=email, password=password)
            st.sidebar.success(f"User {user.email} created successfully")
            return True
        except Exception as e:
            st.sidebar.error(f"Error creating user: {e}")
            return False

def main():
    st.title("Student Attendance Management System")
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if not st.session_state.is_authenticated:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            email = st.text_input("Email", key="login_email_main")
            password = st.text_input("Password", type="password", key="login_password_main")
            if st.button("Login", key="login_button"):
                try:
                    user = auth.get_user_by_email(email)
                    st.session_state.is_authenticated = True
                    st.success(f"Welcome {user.email}")
                    st.rerun()
                except Exception as e:
                    st.error("Invalid credentials")
        with register_tab:
            email = st.text_input("Email", key="register_email_main")
            password = st.text_input("Password", type="password", key="register_password_main")
            if st.button("Register", key="register_button"):
                try:
                    user = auth.create_user(email=email, password=password)
                    st.success(f"User {user.email} created successfully! Please login.")
                except Exception as e:
                    st.error(f"Error creating user: {e}")
    if st.session_state.is_authenticated:
        st.sidebar.title("User Options")
        if st.sidebar.button("Logout"):
            st.session_state.is_authenticated = False
            st.rerun()
        if 'attendance_df' not in st.session_state:
            st.session_state.attendance_df = None
        if 'validated' not in st.session_state:
            st.session_state.validated = False
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        sign_detection_tab, old_results_tab = st.tabs(["Sign Detection", "Old Results"])
        with sign_detection_tab:
            st.header("Sign Detection")
            if st.session_state.current_step == 1:
                st.subheader("Step 1: Select Course and Date")
                courses = [course['name'] for course in courses_collection.find()]
                if not courses:
                    st.warning("No courses available. Please add a course in the Course Management tab.")
                    st.session_state.current_step = 1
                else:
                    course_name = st.selectbox("Select Course", courses)
                    attendance_date = st.date_input("Select Attendance Date")
                    if st.button("Next", key="step1_next"):
                        st.session_state.selected_course = course_name
                        st.session_state.selected_date = attendance_date
                        st.session_state.current_step = 2
                        st.rerun()
            elif st.session_state.current_step == 2:
                st.subheader(f"Step 2: Upload Attendance Sheet for {st.session_state.selected_course} on {st.session_state.selected_date}")
                uploaded_file = st.file_uploader("Upload attendance sheet image", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    st.image(uploaded_file, caption="Uploaded Attendance Sheet", use_column_width=True)
                    attendance_df = extract_info(tmp_path, st.session_state.selected_course, str(st.session_state.selected_date))
                    if attendance_df is not None:
                        st.session_state.attendance_df = attendance_df
                        st.session_state.validated = False
        with old_results_tab:
            st.header("Old Results")
            st.write("### Past Attendance Data")
            past_attendance = list(attendance_collection.find())
            
            if past_attendance:
                for idx, record in enumerate(past_attendance):
                    try:
                        date_obj = datetime.strptime(record['date'], '%Y-%m-%d')
                        formatted_date = date_obj.strftime('%B %d, %Y')
                    except:
                        formatted_date = record['date']
                    
                    st.write(f"**Course:** {record['course_name']}, **Date:** {formatted_date}")
                    
                    if st.button(f"View Details", key=f"view_details_{idx}"):
                        st.write("#### Attendance Details")
                        attendance_data = record.get('attendance', [])
                        
                        # Get validation data
                        validation_record = signature_collection.find_one({
                            'course_name': record['course_name'],
                            'date': record['date']
                        })
                        
                        if attendance_data:
                            df = pd.DataFrame(attendance_data)
                            
                            # Merge validation data if available
                            if validation_record:
                                validation_df = pd.DataFrame(validation_record['validation_results'])
                                df = df.merge(validation_df[['ID', 'Morning_Prediction', 'Morning_Confidence', 
                                                           'Evening_Prediction', 'Evening_Confidence']], 
                                            on='ID', how='left')
                            
                            st.dataframe(df)
                            
                            # Update visualizations to include validation results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                morning_counts = df['Morning'].value_counts()
                                fig1, ax1 = plt.subplots()
                                ax1.pie(morning_counts, labels=morning_counts.index, autopct='%1.1f%%')
                                ax1.set_title('Morning Attendance')
                                st.pyplot(fig1)
                            
                            with col2:
                                evening_counts = df['Evening'].value_counts()
                                fig2, ax2 = plt.subplots()
                                ax2.pie(evening_counts, labels=evening_counts.index, autopct='%1.1f%%')
                                ax2.set_title('Evening Attendance')
                                st.pyplot(fig2)
                            
                            if validation_record:
                                with col3:
                                    validation_counts = df['Morning_Prediction'].value_counts()
                                    fig3, ax3 = plt.subplots()
                                    ax3.pie(validation_counts, labels=validation_counts.index, autopct='%1.1f%%')
                                    ax3.set_title('Signature Validation Results')
                                    st.pyplot(fig3)
                            
                            # Add validation statistics if available
                            if validation_record:
                                st.write("#### Validation Statistics")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Genuine Morning Signatures", 
                                            len(df[df['Morning_Prediction'] == 'genuine']))
                                    st.metric("Average Morning Confidence", 
                                            f"{df['Morning_Confidence'].mean():.2f}")
                                with col2:
                                    st.metric("Genuine Evening Signatures", 
                                            len(df[df['Evening_Prediction'] == 'genuine']))
                                    st.metric("Average Evening Confidence", 
                                            f"{df['Evening_Confidence'].mean():.2f}")
                    
                            # Add visualizations
                            col1, col2 = st.columns(2)
                            with col1:
                                # Morning attendance pie chart
                                morning_counts = df['Morning'].value_counts()
                                fig1, ax1 = plt.subplots()
                                ax1.pie(morning_counts, labels=morning_counts.index, autopct='%1.1f%%')
                                ax1.set_title('Morning Attendance Distribution')
                                st.pyplot(fig1)
                            
                            with col2:
                                # Evening attendance pie chart
                                evening_counts = df['Evening'].value_counts()
                                fig2, ax2 = plt.subplots()
                                ax2.pie(evening_counts, labels=evening_counts.index, autopct='%1.1f%%')
                                ax2.set_title('Evening Attendance Distribution')
                                st.pyplot(fig2)
                            
                            # Bar chart for both sessions
                            st.write("#### Attendance Comparison")
                            attendance_comparison = pd.DataFrame({
                                'Morning': morning_counts,
                                'Evening': evening_counts
                            })
                            st.bar_chart(attendance_comparison)
                            
                            # Display signature gallery for this record
                            st.write("#### Signature Gallery")
                            morning_tab, evening_tab = st.tabs(["Morning Signatures", "Evening Signatures"])
                            
                            course_date_path = os.path.join('signatures', record['course_name'], str(record['date']))
                            
                            with morning_tab:
                                cols = st.columns(3)
                                col_idx = 0
                                for _, row in df.iterrows():
                                    student_id = row['ID']
                                    morning_sig_path = None
                                    
                                    if os.path.exists(course_date_path):
                                        for filename in os.listdir(course_date_path):
                                            if student_id in filename and "morning" in filename:
                                                morning_sig_path = os.path.join(course_date_path, filename)
                                                break
                                    
                                    if morning_sig_path and os.path.exists(morning_sig_path):
                                        with cols[col_idx]:
                                            img = Image.open(morning_sig_path)
                                            st.image(img, caption=f"{row['Name']} ({student_id})")
                                            if 'Morning_Prediction' in row and row['Morning_Prediction']:
                                                prediction = row['Morning_Prediction']
                                                confidence = row['Morning_Confidence']
                                                color = "green" if prediction == "genuine" else "red"
                                                st.markdown(f"""
                                                    <div style='color:{color};'>
                                                        <p>Status: {row['Morning']}</p>
                                                        <p>Validation: {prediction}</p>
                                                        <p>Confidence: {confidence:.4f}</p>
                                                    </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                st.write(f"Status: {row['Morning']}")
                                        col_idx = (col_idx + 1) % 3
                            
                            with evening_tab:
                                cols = st.columns(3)
                                col_idx = 0
                                for _, row in df.iterrows():
                                    student_id = row['ID']
                                    evening_sig_path = None
                                    
                                    if os.path.exists(course_date_path):
                                        for filename in os.listdir(course_date_path):
                                            if student_id in filename and "evening" in filename:
                                                evening_sig_path = os.path.join(course_date_path, filename)
                                                break
                                    
                                    if evening_sig_path and os.path.exists(evening_sig_path):
                                        with cols[col_idx]:
                                            img = Image.open(evening_sig_path)
                                            st.image(img, caption=f"{row['Name']} ({student_id})")
                                            if 'Evening_Prediction' in row and row['Evening_Prediction']:
                                                prediction = row['Evening_Prediction']
                                                confidence = row['Evening_Confidence']
                                                color = "green" if prediction == "genuine" else "red"
                                                st.markdown(f"""
                                                    <div style='color:{color};'>
                                                        <p>Status: {row['Evening']}</p>
                                                        <p>Validation: {prediction}</p>
                                                        <p>Confidence: {confidence:.4f}</p>
                                                    </div>
                                                """, unsafe_allow_html=True)
                                            else:
                                                st.write(f"Status: {row['Evening']}")
                                        col_idx = (col_idx + 1) % 3
                            
                            # Add download button for this record
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"attendance_{record['course_name']}_{record['date']}.csv",
                                mime="text/csv",
                                key=f"download_{idx}"
                            )
                        else:
                            st.warning("No attendance data available for this record")
            else:
                st.warning("No past attendance data found.")
        if st.session_state.attendance_df is not None:
            model_file = st.file_uploader("Upload signature verification model (or use default)", type=["h5"])
            default_model_path = '/Users/leopard/Desktop/Cource_Work_2/signature/APP/student-attendance-management/best_signature_model.h5'
            model_path = default_model_path
            if model_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    tmp_file.write(model_file.getvalue())
                    model_path = tmp_file.name
                st.success(f"Model uploaded successfully to {model_path}")
            else:
                if os.path.exists(default_model_path):
                    st.info(f"Using default model at {default_model_path}")
                else:
                    st.warning("Default model not found. Please upload a model file.")
            if st.button("Validate Signatures"):
                validated_df = validate_signatures(st.session_state.attendance_df, model_path, st.session_state.selected_course, str(st.session_state.selected_date))
                st.session_state.attendance_df = validated_df
                st.session_state.validated = True
                st.success("Signature validation completed!")
                st.write("### Validation Results")
                st.dataframe(validated_df)
                validated_df.to_csv('validated_attendance.csv', index=False)
                st.success("Validated data saved to validated_attendance.csv")
                csv = validated_df.to_csv(index=False)
                st.download_button(
                    label="Download Validated CSV",
                    data=csv,
                    file_name="validated_attendance.csv",
                    mime="text/csv"
                )
        if st.session_state.validated:
            display_signature_gallery(
                st.session_state.attendance_df,
                st.session_state.selected_course,
                str(st.session_state.selected_date)
            )
            if st.button("Complete and Clear"):
                st.session_state.attendance_df = None
                st.session_state.validated = False
                st.session_state.current_step = 1
                st.session_state.selected_course = None
                st.session_state.selected_date = None
                st.success("All data cleared. Ready for new attendance session!")
                st.rerun()
        st.sidebar.header("Course Management")
        new_course_name = st.sidebar.text_input("Add New Course")
        if st.sidebar.button("Add Course"):
            if new_course_name:
                courses_collection.insert_one({'name': new_course_name})
                st.sidebar.success(f"Course '{new_course_name}' added successfully!")
            else:
                st.sidebar.error("Please enter a course name")

if __name__ == "__main__":
    main()

# > streamlit run app.py