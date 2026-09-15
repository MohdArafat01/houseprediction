# Real Estate Valuation Engine (BCA 5th Sem Minor Project)

An end-to-end Machine Learning web service built to estimate residential property valuations based on physical characteristics and location sector parameters.

## Core Features
- Multi-variable Regression using Random Forest Ensembles.
- Dynamic One-Hot Encoding for categorical geographical sectors.
- Responsive HTML5/CSS3 Dashboard with Async Fetch REST API integration.

---

## 🧠 Machine Learning Architecture & Technical Specifications

This application utilizes a supervised machine learning regression pipeline built on top of the **Scikit-Learn** ecosystem. Below is a detailed breakdown of the ML methodologies, feature transformations, and algorithms employed in the project.

### 1. Feature Engineering & Preprocessing
To prepare the dataset for the regression pipeline, raw data undergoes a sequence of mathematical transformations:

* **Categorical Encoding (One-Hot Encoding)**:
  * **Module Used**: `pandas.get_dummies()`
  * **Purpose**: Categorical values (such as `Location Sector`) are converted into sparse binary vector representations ($0$ or $1$) so the model can process geographical weights numerically without introducing artificial ordinal relationships.

* **Feature Standardization**:
  * **Module Used**: `sklearn.preprocessing.StandardScaler`
  * **Mathematical Formula**:
    $$Z = \frac{x - \mu}{\sigma}$$
    *(where $x$ is the feature value, $\mu$ is the mean, and $\sigma$ is the standard deviation)*
  * **Purpose**: Features like `sqft` (ranging from 600–4500) operate on a significantly higher magnitude than `bedrooms` (ranging from 1–5). Standardizing features to a mean ($\mu = 0$) and unit variance ($\sigma = 1$) prevents high-magnitude features from dominating tree splits and distance metrics.

---

### 2. Core Algorithm: Random Forest Regressor
* **Module Used**: `sklearn.ensemble.RandomForestRegressor`
* **Algorithm Type**: Supervised Learning (Ensemble Bagging Technique)
* **Hyperparameters Configured**:
  * `n_estimators=150`: Constructs an ensemble of 150 independent decision trees.
  * `max_depth=12`: Limits tree depth to prevent overfitting and ensure high generalization on unseen data.
  * `random_state=42`: Fixes the pseudo-random seed to guarantee reproducible results across training runs.

#### **How It Works Under the Hood**:
1. **Bootstrap Aggregating (Bagging)**: The algorithm selects multiple random subsets of data (with replacement) to train each individual decision tree independently.
2. **Feature Subsampling**: At each node split in a decision tree, only a random subset of features is considered, reducing correlation between individual trees.
3. **Ensemble Averaging**: Final property value predictions are calculated by aggregating and averaging the prediction output ($\hat{y}$) across all 150 individual decision trees:
   $$\hat{y} = \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$

---

### 3. Model Evaluation Metrics
The model is evaluated using standard regression statistics to verify predictive reliability:

* **Coefficient of Determination ($R^2$ Score)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  * **Result**: `~0.965` (Explains ~96.5% of the variance in housing prices).

* **Mean Absolute Error (MAE)**:
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  * **Result**: Measures average deviation between predicted values and actual ground truth prices in dollars.

---

### 4. Model Persistence Pipeline
* **Module Used**: `joblib`
* Trained model weights (`model.pkl`), feature scaling parameters (`scaler.pkl`), and column schemas (`columns.pkl`) are serialized directly to disk.
* The Flask REST API deserializes these binaries into memory during startup, enabling sub-millisecond inference times without re-training the model per request.

## Execution Guide

1. Install Dependencies:
   pip install -r requirements.txt

2. Train Model & Export Artifacts:
   python train.py

3. Run Flask REST API:
   python app.py

4. Access Web Client:
   Open browser at http://127.0.0.1:5000
   