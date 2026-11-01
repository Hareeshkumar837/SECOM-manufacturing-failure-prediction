# SECOM-manufacturing-failure-prediction
An experimental machine learning project using the UCI SECOM dataset to predict semiconductor manufacturing failures from high-dimensional sensor data.

🛠 Tech Stack

Python: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib

⚙️ Workflow

Data Preprocessing: Handled missing values and applied RobustScaler.

Imbalance Handling: Applied SMOTE for minority class oversampling.

Classification: Trained a Logistic Regression baseline model tuned with GridSearchCV.

Evaluation: Evaluated performance focused on Precision, Recall, and Average Precision.

📊 Baseline Results

Accuracy: 52%

Positive Recall: 0.71

Positive Precision: 0.09

Average Precision: 0.131

Note: The current model detects most failures (71% recall) but produces a high number of false positives.

📂 Dataset

Source: UCI SECOM Dataset

