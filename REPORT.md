# CIFAR-10 Classification with K-Nearest Neighbors (KNN)

This project applies a K-Nearest Neighbors (KNN) classifier to the CIFAR-10 image dataset.  
The goal was to evaluate how well a simple distance-based model can classify images under different data split strategies and to observe how the choice of *k* affects performance.


## 1. Experimental Setup

- **Image preprocessing**
  - Resized to 32×32 RGB  
  - Flattened into 3072-dimensional vectors  
  - Pixel values normalized to [0, 1]

- **Model configuration**
  - Distance metric: Euclidean  
  - Weights: uniform  
  - Random seed: 42  
  - Implemented using `scikit-learn`’s `KNeighborsClassifier`

- **Data splitting**
  - Train/Test split: 80% / 20%  
  - Train/Validation/Test split: 64% / 16% / 20%  
  - 5-Fold Cross-Validation: stratified and shuffled (random_state=42)


## 2. Train/Test Split (k = 5)

| Metric | Score |
|--------|--------|
| Accuracy | **0.329** |
| Precision (macro) | 0.419 |
| Recall (macro) | 0.329 |
| F1-score (macro) | 0.313 |

**Notes**  
Classes with distinct colors or backgrounds (such as *airplane* and *ship*) were recognized relatively well,  
while visually similar categories (*cat*, *dog*, *automobile*, *truck*) were often confused.  
Using raw pixel vectors without spatial or texture information limited the model’s accuracy.


## 3. Train/Validation/Test (Hyperparameter Search)

**Validation accuracy per k:**

(1, 0.3319), (3, 0.3121), (5, 0.3204), (7, 0.3220), (9, 0.3191), (11, 0.3193), ...

- **Chosen k:** 1 (highest validation accuracy = 0.3319)

**Final Test Results (k = 1)**

| Metric | Score |
|--------|--------|
| Accuracy | **0.3389** |
| Precision (macro) | 0.3936 |
| Recall (macro) | 0.3389 |
| F1-score (macro) | 0.3302 |

**Notes**  
The validation experiment showed that smaller *k* values performed better, and *k=1* achieved the highest validation accuracy.  
Using only the nearest neighbor made the model sensitive to noise but better at capturing fine color variations.  
Compared to the initial train/test split, the test accuracy slightly improved from 0.329 to 0.3389.


## 4. 5-Fold Cross-Validation

| k | Mean Accuracy | Std. Dev. |
|--:|:--------------:|:----------:|
| **1** | **0.3404** | **± 0.0020** |
| 3 | 0.3236 | ± 0.0027 |
| 5 | 0.3327 | ± 0.0025 |
| 7 | 0.3315 | ± 0.0025 |
| 9 | 0.3335 | ± 0.0022 |
| ... | ... | ... |

**Notes**  
Cross-validation confirmed that *k=1* consistently achieved the best performance (0.3404 ± 0.0020).  
Accuracy gradually decreased as *k* increased, showing that larger neighborhoods tended to underfit.  
The small standard deviation indicates stable results across different folds.  
The plot of these results is saved as `results/cv_accuracy_vs_k.png`.

## 5. Discussion

These results show that using flattened RGB pixel vectors limits performance because no spatial or edge information is retained.  
Color-dominated categories (*ship*, *airplane*) were easier to distinguish, while shape-based categories (*cat*, *dog*) remained difficult.  

Possible improvements:
1. **Color histograms** for richer color features  
2. **HOG (Histogram of Oriented Gradients)** for edge and texture representation  
3. **PCA** for dimensionality reduction and noise filtering  


## 6. Reproducibility

# 1. Basic train/test experiment
python src/knn.py --mode train_test --k 5

# 2. Hyperparameter tuning
python src/knn.py --mode train_val_test --k_min 1 --k_max 29 --k_step 2

# 3. 5-fold cross-validation
python src/knn.py --mode cv --k_min 1 --k_max 29 --k_step 2

# 4. Final submission
python src/knn.py --mode submit --k 1
Environment

Python 3.8 (Homebrew + venv)

NumPy 1.24.4

Pandas 2.0.3

Scikit-learn 1.3.2

Pillow 10.4.0

Matplotlib 3.7.5



## 7. Reproducibility

This project showed that feature design and preprocessing can influence image classification more than the model itself.
Although KNN is simple, its performance depends heavily on how input data are represented.
Cross-validation also highlighted how different splits can affect results, reinforcing the importance of reproducible experimentation.