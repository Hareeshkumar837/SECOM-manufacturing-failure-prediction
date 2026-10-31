import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,average_precision_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv(Path(__file__).parent/'uci-secom.csv')

df = df.drop(columns=['Time'])

x = df.drop(columns=['Pass/Fail'])
y = df['Pass/Fail']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',RobustScaler()),
    ('smote',SMOTE(random_state=42)),
    ('classifier',LogisticRegression(class_weight='balanced',solver='saga',max_iter=12000,random_state=42))
])

param_grid = {
    'imputer__strategy':['median','mean'],
    'smote__sampling_strategy':[0.8],
    'classifier__C':[0.1,1.0],
    'classifier__l1_ratio':[0,0.5,1]
    
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='average_precision',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train,y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(x_test)
y_proba = best_model.predict_proba(x_test)[:, 1]



pr_auc = average_precision_score(y_test,y_proba)

print(f"Final PR_AUC Score : {pr_auc:.3f}")
print("Classification report: \n",classification_report(y_test,y_pred))

classifier_step = best_model.named_steps['classifier']
coefficients = classifier_step.coef_[0]
feature_name = x.columns

feature_importance = pd.DataFrame({
    'Sensor':feature_name,
    'Importance':coefficients,
    'Absolute_imp':np.abs(coefficients)


}).sort_values(by='Absolute_imp',ascending=False)

print("\n--- TOP 5 SENSORS DRIVING FAILURES ---")
print(feature_importance.head(5)[['Sensor', 'Importance']])


cm = confusion_matrix(y_test,y_pred)
labels=[-1,1]

print('Confusion matrix')
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Pass(-1)","Fail(1)"]
)

disp.plot()
plt.show()


