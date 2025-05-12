import os
import numpy as np
import cv2
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from tqdm import tqdm
from skimage.feature import hog

DATASET_PATH = r"C:\Users\ASUS\Desktop\VS code\Web\SOLO\AgriBott\model\plantvillage dataset\color"
MODEL_DIR = r"C:\Users\ASUS\Desktop\VS code\Web\SOLO\AgriBott\simple\model"
os.makedirs(MODEL_DIR, exist_ok=True)

def extract_features(img, img_size=96):
    """
    Extract HOG and color statistics features from image.
    """
    img = cv2.resize(img, (img_size, img_size))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(12, 12),
                       cells_per_block=(2, 2), visualize=False)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_mean, h_std = np.mean(hsv_img[:,:,0]), np.std(hsv_img[:,:,0])
    s_mean, s_std = np.mean(hsv_img[:,:,1]), np.std(hsv_img[:,:,1])
    v_mean, v_std = np.mean(hsv_img[:,:,2]), np.std(hsv_img[:,:,2])
    r_mean, r_std = np.mean(img[:,:,0]), np.std(img[:,:,0])
    g_mean, g_std = np.mean(img[:,:,1]), np.std(img[:,:,1])
    b_mean, b_std = np.mean(img[:,:,2]), np.std(img[:,:,2])
    color_features = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std,
                               r_mean, r_std, g_mean, g_std, b_mean, b_std])
    combined_features = np.concatenate((hog_features, color_features))
    return combined_features

def load_dataset(dataset_path, img_size=96, test_size=0.2, limit_per_class=180):
    print("Loading and preprocessing dataset...")
    X, y, classes = [], [], []
    for idx, class_dir in enumerate(sorted(os.listdir(dataset_path))):
        class_path = os.path.join(dataset_path, class_dir)
        if not os.path.isdir(class_path):
            continue
        print(f"Processing class: {class_dir}")
        classes.append(class_dir)
        img_files = sorted(os.listdir(class_path))
        if limit_per_class:
            img_files = img_files[:limit_per_class]
        for img_file in tqdm(img_files, desc=f"Class {idx+1}/{len(os.listdir(dataset_path))}"):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                features = extract_features(img, img_size)
                X.append(features)
                y.append(idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(f"Feature extraction complete. Features shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"Dataset loaded: {len(X)} images across {len(classes)} classes")
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    return X_train, X_test, y_train, y_test, classes

def reduce_dimensionality(X_train, X_test, n_components=200):
    print(f"Reducing feature dimensionality to {n_components} with PCA...")
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.2f}")
    return X_train_pca, X_test_pca, pca

def train_model(X_train, y_train):
    print("Training HistGradientBoostingClassifier...")
    start_time = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=12,
        learning_rate=0.15,
        l2_regularization=0.2,
        early_stopping=True,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time() - start_time:.2f} seconds")
    return clf

def evaluate_model(clf, X_test, y_test, class_names):
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    return accuracy, y_pred

def save_model(clf, pca, class_names, output_path):
    print(f"Saving model to {output_path}...")
    model_data = {
        'classifier': clf,
        'pca': pca,
        'classes': class_names
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model saved. File size: {model_size:.2f} MB")
    classes_path = os.path.join(os.path.dirname(output_path), 'class_names.pkl')
    with open(classes_path, 'wb') as f:
        pickle.dump(class_names, f)

def plot_confusion_matrix(y_test, y_pred, class_names, output_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path, dpi=100)
    print(f"Confusion matrix saved to {output_path}")

def main():
    print("Starting Plant Disease Detection model training (compact & accurate)...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please download the PlantVillage dataset and extract it to this location.")
        return
    X_train, X_test, y_train, y_test, class_names = load_dataset(
        DATASET_PATH,
        img_size=96,
        test_size=0.2,
        limit_per_class=180  # Reduce for smaller model
    )
    X_train_pca, X_test_pca, pca = reduce_dimensionality(X_train, X_test, n_components=200)
    model = train_model(X_train_pca, y_train)
    accuracy, y_pred = evaluate_model(model, X_test_pca, y_test, class_names)
    model_path = os.path.join(MODEL_DIR, 'plant_disease_model_compact.pkl')
    save_model(model, pca, class_names, model_path)
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    print("Training complete!")

if __name__ == "__main__":
    main()