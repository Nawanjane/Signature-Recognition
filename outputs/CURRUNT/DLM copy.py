import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import datetime
from pathlib import Path

# Create output directories
output_dir = '/Users/leopard/Desktop/Cource_Work_2/outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)

# Set up logging
log_file = os.path.join(output_dir, 'signature_verification_log.txt')
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

log_message("Starting signature verification process")

# Set the paths to the dataset
base_path = 'data/Dataset_Signature_Final'
genuine_path = os.path.join(base_path, 'real')
forged_path = os.path.join(base_path, 'forge')

# Check if the paths exist
log_message(f"Genuine path exists: {os.path.exists(genuine_path)}")
log_message(f"Forged path exists: {os.path.exists(forged_path)}")

# Function to get all image files from a directory
def get_image_files(directory):
    if not os.path.exists(directory):
        log_message(f"Directory {directory} does not exist!")
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

# Get all image files
genuine_images = get_image_files(genuine_path)
forged_images = get_image_files(forged_path)

log_message(f"Number of genuine signature images: {len(genuine_images)}")
log_message(f"Number of forged signature images: {len(forged_images)}")

# Create dataframes for genuine and forged signatures
genuine_data = []
for img_path in genuine_images:
    base_name = os.path.basename(img_path)
    genuine_data.append({
        'filename': base_name,
        'path': img_path,
        'type': 'real'
    })

forged_data = []
for img_path in forged_images:
    base_name = os.path.basename(img_path)
    forged_data.append({
        'filename': base_name,
        'path': img_path,
        'type': 'forge'
    })

# Create dataframes
genuine_df = pd.DataFrame(genuine_data)
forged_df = pd.DataFrame(forged_data)

# Combine the dataframes
all_signatures_df = pd.concat([genuine_df, forged_df], ignore_index=True)

# Save the dataframes
genuine_df.to_csv(os.path.join(output_dir, 'genuine_signatures.csv'), index=False)
forged_df.to_csv(os.path.join(output_dir, 'forged_signatures.csv'), index=False)
all_signatures_df.to_csv(os.path.join(output_dir, 'all_signatures.csv'), index=False)

# Display the first few rows
log_message("\nSample of signature data:")
log_message(str(all_signatures_df.head()))

# Basic statistics
log_message("\nBasic statistics:")
log_message(f"Total number of signatures: {len(all_signatures_df)}")
log_message(f"Number of genuine signatures: {len(genuine_df)}")
log_message(f"Number of forged signatures: {len(forged_df)}")

# Display sample images from both genuine and forged categories
def display_sample_images(genuine_df, forged_df, num_samples=3, save_path=None):
    plt.figure(figsize=(15, 10))
    
    # Display genuine signatures
    for i in range(min(num_samples, len(genuine_df))):
        img_path = genuine_df.iloc[i]['path']
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, num_samples, i+1)
            plt.imshow(img)
            plt.title(f"Genuine: {genuine_df.iloc[i]['filename']}")
            plt.axis('off')
    
    # Display forged signatures
    for i in range(min(num_samples, len(forged_df))):
        img_path = forged_df.iloc[i]['path']
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(img)
            plt.title(f"Forged: {forged_df.iloc[i]['filename']}")
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        log_message(f"Sample images saved to {save_path}")
    
    plt.show()

# Display and save sample images
log_message("\nSample Signature Images:")
sample_images_path = os.path.join(output_dir, 'figures', 'sample_signatures.png')
display_sample_images(genuine_df, forged_df, save_path=sample_images_path)

# Function to load and preprocess images
def load_images(dataframe, label_column='type', target_size=(128, 128)):
    images = []
    labels = []
    
    for _, row in dataframe.iterrows():
        try:
            # Read image
            img = cv2.imread(row['path'])
            if img is None:
                log_message(f"Failed to load image: {row['path']}")
                continue
                
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            img = cv2.resize(img, target_size)
            
            # Normalize pixel values to [0, 1]
            img = img / 255.0
            
            # Add channel dimension (required for CNN)
            img = np.expand_dims(img, axis=-1)
            
            # Add to dataset
            images.append(img)
            # Fix: Use 'real' instead of 'genuine' to match your dataframe
            labels.append(1 if row[label_column] == 'real' else 0)
            
        except Exception as e:
            log_message(f"Error processing {row['path']}: {e}")
    
    return np.array(images), np.array(labels)

# Load and prepare the dataset
log_message("Loading and preprocessing images...")
X, y = load_images(all_signatures_df)
log_message(f"Dataset loaded: {X.shape[0]} images, {X.shape[1]}x{X.shape[2]} pixels")

# Save sample preprocessed images
def save_preprocessed_samples(X, y, num_samples=3):
    plt.figure(figsize=(15, 10))
    
    # Get indices of genuine and forged samples
    genuine_indices = np.where(y == 1)[0]
    forged_indices = np.where(y == 0)[0]
    
    # Display genuine samples
    for i in range(min(num_samples, len(genuine_indices))):
        plt.subplot(2, num_samples, i+1)
        plt.imshow(X[genuine_indices[i]].squeeze(), cmap='gray')
        plt.title(f"Genuine (Preprocessed)")
        plt.axis('off')
    
    # Display forged samples
    for i in range(min(num_samples, len(forged_indices))):
        plt.subplot(2, num_samples, num_samples+i+1)
        plt.imshow(X[forged_indices[i]].squeeze(), cmap='gray')
        plt.title(f"Forged (Preprocessed)")
        plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'figures', 'preprocessed_samples.png')
    plt.savefig(save_path)
    log_message(f"Preprocessed samples saved to {save_path}")
    plt.show()

save_preprocessed_samples(X, y)

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

log_message(f"Training set: {X_train.shape[0]} images")
log_message(f"Validation set: {X_val.shape[0]} images")
log_message(f"Test set: {X_test.shape[0]} images")

# Calculate class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
log_message(f"Class weights: {class_weight_dict}")

# Remove data augmentation and visualization code
log_message("Skipping data augmentation as requested")

# Build the CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        # First convolutional block with batch normalization
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Second convolutional block with batch normalization
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third convolutional block with batch normalization
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flatten and fully connected layers with stronger regularization
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: genuine or forged
    ])
    
    # Compile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Create and compile the model
input_shape = X_train[0].shape
model = build_cnn_model(input_shape)

# Save model summary
model_summary_path = os.path.join(output_dir, 'reports', 'model_summary.txt')
with open(model_summary_path, 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
log_message(f"Model summary saved to {model_summary_path}")

# Custom F1 Score callback
class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_f1 = 0
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred_prob = self.model.predict(x_val)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred)
        logs['val_f1'] = f1
        log_message(f"Epoch {epoch+1}: val_f1 = {f1:.4f}")
        
        if f1 > self.best_f1:
            self.best_f1 = f1

# Add learning rate scheduler
def lr_schedule(epoch):
    initial_lr = 0.00005
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.5
    else:
        return initial_lr * 0.1

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Set up callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(os.path.join(output_dir, 'models', 'best_signature_model.h5'), 
                    monitor='val_loss', 
                    save_best_only=True, 
                    mode='min'),
    F1ScoreCallback(validation_data=(X_val, y_val)),
    lr_scheduler
]

# Train the model
log_message("Starting model training...")
batch_size = 16  # Smaller batch size
epochs = 30      # Fewer epochs

# Train without data augmentation
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(output_dir, 'reports', 'training_history.csv'), index=False)
log_message("Training history saved to CSV")

# Plot training history
plt.figure(figsize=(15, 10))

# Plot training & validation accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation precision
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
training_plot_path = os.path.join(output_dir, 'figures', 'training_history.png')
plt.savefig(training_plot_path)
log_message(f"Training history plot saved to {training_plot_path}")
plt.show()

# Find the optimal threshold
log_message("Finding optimal decision threshold...")
y_pred_prob = model.predict(X_test)
thresholds = np.arange(0.3, 0.7, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_prob > threshold).astype(int).flatten()
    report = classification_report(y_test, y_pred_threshold, target_names=['Forged', 'Genuine'], output_dict=True)
    f1_genuine = report['Genuine']['f1-score']
    precision_genuine = report['Genuine']['precision']
    recall_genuine = report['Genuine']['recall']
    
    threshold_results.append({
        'threshold': threshold,
        'f1_genuine': f1_genuine,
        'precision_genuine': precision_genuine,
        'recall_genuine': recall_genuine
    })
    
    log_message(f"Threshold: {threshold:.2f}, F1-score for Genuine: {f1_genuine:.4f}")

# Save threshold results
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(os.path.join(output_dir, 'reports', 'threshold_analysis.csv'), index=False)

# Find best threshold based on F1 score
best_threshold_row = threshold_df.loc[threshold_df['f1_genuine'].idxmax()]
best_threshold = best_threshold_row['threshold']
best_f1 = best_threshold_row['f1_genuine']
log_message(f"\nBest threshold: {best_threshold:.2f} with F1-score: {best_f1:.4f}")

# Plot threshold analysis
plt.figure(figsize=(10, 6))
plt.plot(threshold_df['threshold'], threshold_df['precision_genuine'], 'b-', label='Precision')
plt.plot(threshold_df['threshold'], threshold_df['recall_genuine'], 'g-', label='Recall')
plt.plot(threshold_df['threshold'], threshold_df['f1_genuine'], 'r-', label='F1 Score')
plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
plt.title('Threshold Analysis for Genuine Class')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
threshold_plot_path = os.path.join(output_dir, 'figures', 'threshold_analysis.png')
plt.savefig(threshold_plot_path)
log_message(f"Threshold analysis plot saved to {threshold_plot_path}")
plt.show()

# Apply the best threshold
y_pred = (y_pred_prob > best_threshold).astype(int).flatten()

# Generate classification report
log_message("\nClassification Report with optimal threshold:")
class_report = classification_report(y_test, y_pred, target_names=['Forged', 'Genuine'])
log_message(class_report)

# Save classification report
with open(os.path.join(output_dir, 'reports', 'classification_report.txt'), 'w') as f:
    f.write(f"Classification Report with threshold {best_threshold:.2f}:\n")
    f.write(class_report)

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix (threshold={best_threshold:.2f})')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Forged', 'Genuine'], rotation=45)
plt.yticks(tick_marks, ['Forged', 'Genuine'])

# Add text annotations to the confusion matrix
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm_plot_path = os.path.join(output_dir, 'figures', 'confusion_matrix.png')
plt.savefig(cm_plot_path)
log_message(f"Confusion matrix saved to {cm_plot_path}")
plt.show()

# Display example predictions
def plot_example_predictions(X, y_true, y_pred, y_pred_prob, num_examples=3):
    # Find examples of each category (TP, TN, FP, FN)
    true_positive = np.where((y_true == 1) & (y_pred == 1))[0]
    true_negative = np.where((y_true == 0) & (y_pred == 0))[0]
    false_positive = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negative = np.where((y_true == 1) & (y_pred == 0))[0]
    
    categories = [
        ('True Positive (Genuine correctly identified)', true_positive),
        ('True Negative (Forged correctly identified)', true_negative),
        ('False Positive (Forged misidentified as Genuine)', false_positive),
        ('False Negative (Genuine misidentified as Forged)', false_negative)
    ]
    
    plt.figure(figsize=(15, 12))
    
    for i, (title, indices) in enumerate(categories):
        if len(indices) > 0:
            for j in range(min(num_examples, len(indices))):
                if j < len(indices):
                    idx = indices[j]
                    plt.subplot(4, num_examples, i*num_examples + j + 1)
                    plt.imshow(X[idx].squeeze(), cmap='gray')
                    plt.title(f"{title}\nProb: {y_pred_prob[idx][0]:.2f}")
                    plt.axis('off')
    
    plt.tight_layout()
    examples_path = os.path.join(output_dir, 'figures', 'example_predictions.png')
    plt.savefig(examples_path)
    log_message(f"Example predictions saved to {examples_path}")
    plt.show()

# Plot example predictions
log_message("\nPlotting example predictions:")
plot_example_predictions(X_test, y_test, y_pred, y_pred_prob)

# Save the final model
final_model_path = os.path.join(output_dir, 'models', 'signature_verification_model.h5')
model.save(final_model_path)
log_message(f"Final model saved to: {final_model_path}")

# Create a function for making predictions on new images
def predict_signature(image_path, model, threshold=0.5, target_size=(128, 128)):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to load image"
            
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Add batch and channel dimensions
        img = np.expand_dims(np.expand_dims(img, axis=0), axis=-1)
        
        # Make prediction
        pred_prob = model.predict(img)[0][0]
        pred_class = 'Genuine' if pred_prob > threshold else 'Forged'
        
        return pred_class, pred_prob
        
    except Exception as e:
        return None, f"Error processing image: {e}"

# Create a demo prediction function
def demo_prediction(model, image_path=None):
    if image_path is None:
        # Use a random test image if no path is provided
        test_indices = np.random.choice(len(X_test), 1)[0]
        test_image = X_test[test_indices]
        true_label = 'Genuine' if y_test[test_indices] == 1 else 'Forged'
        
        # Save the test image to a temporary file
        temp_image_path = os.path.join(output_dir, 'temp_test_image.png')
        plt.imsave(temp_image_path, test_image.squeeze(), cmap='gray')
        image_path = temp_image_path
    
        # Make prediction
        pred_class, pred_prob = predict_signature(image_path, model, best_threshold)
        
        # Display results
        plt.figure(figsize=(8, 6))
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}, Predicted: {pred_class} (Prob: {pred_prob:.4f})")
        plt.axis('off')
        plt.tight_layout()
        demo_path = os.path.join(output_dir, 'figures', 'demo_prediction.png')
        plt.savefig(demo_path)
        log_message(f"Demo prediction saved to {demo_path}")
        plt.show()
        
        log_message(f"True label: {true_label}")
        log_message(f"Predicted: {pred_class} with probability {pred_prob:.4f}")
    else:
        # Make prediction on the provided image
        pred_class, pred_prob = predict_signature(image_path, model, best_threshold)
        
        # Display results
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {pred_class} (Prob: {pred_prob:.4f})")
        plt.axis('off')
        plt.tight_layout()
        demo_path = os.path.join(output_dir, 'figures', 'custom_prediction.png')
        plt.savefig(demo_path)
        log_message(f"Custom prediction saved to {demo_path}")
        plt.show()
        
        log_message(f"Predicted: {pred_class} with probability {pred_prob:.4f}")

# Run a demo prediction
log_message("\nRunning demo prediction:")
demo_prediction(model)

# Generate a summary report
summary_report = f"""
# Signature Verification Model Summary Report

## Dataset Information
- Total signatures: {len(all_signatures_df)}
- Genuine signatures: {len(genuine_df)}
- Forged signatures: {len(forged_df)}

## Model Architecture
- CNN with 3 convolutional blocks
- Input shape: {input_shape}
- Regularization: L2 and Dropout

## Training Information
- Training samples: {len(X_train)}
- Validation samples: {len(X_val)}
- Test samples: {len(X_test)}
- Batch size: {batch_size}
- Class weights: {class_weight_dict}

## Performance Metrics
- Best threshold: {best_threshold:.2f}
- Test accuracy: {classification_report(y_test, y_pred, output_dict=True)['accuracy']:.4f}
- F1 score (Genuine): {classification_report(y_test, y_pred, output_dict=True)['Genuine']['f1-score']:.4f}
- Precision (Genuine): {classification_report(y_test, y_pred, output_dict=True)['Genuine']['precision']:.4f}
- Recall (Genuine): {classification_report(y_test, y_pred, output_dict=True)['Genuine']['recall']:.4f}

## Files Generated
- Model: {final_model_path}
- Training history: {os.path.join(output_dir, 'reports', 'training_history.csv')}
- Classification report: {os.path.join(output_dir, 'reports', 'classification_report.txt')}
- Figures: {os.path.join(output_dir, 'figures')}

## Timestamp
- {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# Save summary report
with open(os.path.join(output_dir, 'reports', 'summary_report.md'), 'w') as f:
    f.write(summary_report)
log_message("Summary report saved.")