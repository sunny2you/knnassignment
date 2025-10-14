 CIFAR-10 KNN Classifier

This repository contains an implementation of a **K-Nearest Neighbors (KNN)** classifier for the **CIFAR-10 image dataset**.  
The goal is to evaluate classification performance under different data split strategies and to analyze how the hyperparameter *k* affects model accuracy.


## 📂 Project Structure

knnassignment/
├─ src/
│ └─ knn.py # Main KNN experiment script
├─ data/
│ ├─ train/ # Training images (id.png)
│ ├─ test/ # Test images (id.png)
│ ├─ trainLabels.csv # Image labels
│ └─ sampleSubmission.csv
├─ results/ # Output metrics and plots
│ ├─ metrics_train_test.json
│ ├─ metrics_train_val_test.json
│ ├─ cv_accuracy_vs_k.png
│ └─ cv_accuracy_vs_k.csv
├─ submission.csv # Final model predictions
├─ REPORT.md 
└─ requirements.txt # Python dependencies


## ⚙️ Requirements

- Python 3.8+
- NumPy 1.24.4  
- Pandas 2.0.3  
- Scikit-learn 1.3.2  
- Pillow 10.4.0  
- Matplotlib 3.7.5  

Install all dependencies:

pip install -r requirements.txt

🚀 How to Run
Run the following commands in order:
# 1. Basic train/test experiment (k=5)
python src/knn.py --mode train_test --k 5

# 2. Train/validation/test split (find best k)
python src/knn.py --mode train_val_test --k_min 1 --k_max 29 --k_step 2

# 3. 5-fold cross-validation
python src/knn.py --mode cv --k_min 1 --k_max 29 --k_step 2

# 4. Generate final submission file (use chosen k)
python src/knn.py --mode submit --k 1


📊 Output Files
File	Description
results/metrics_train_test.json	Metrics from simple train/test split
results/metrics_train_val_test.json	Validation scores and selected k
results/cv_accuracy_vs_k.png	Cross-validation accuracy plot
submission.csv	Final predictions for test data

🧠 Summary
Images were resized to 32×32 RGB and flattened into 3072-dimensional vectors.

Euclidean distance and uniform weights were used for KNN.

Best validation result obtained at k=1 with test accuracy 0.3389.

Cross-validation confirmed stable performance (0.3404 ± 0.0020).
