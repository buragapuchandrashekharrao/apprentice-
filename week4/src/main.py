"""
Week 4 — Machine Learning Model Development and Evaluation
Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.

Target:
0 = malignant
1 = benign

Model:
StandardScaler + LogisticRegression

The pipeline includes:
1. Dataset loading
2. Stratified train/test split
3. Feature scaling
4. Model training
5. Classification metrics
6. Confusion matrix
7. ROC curve and AUC
8. Feature coefficient inspection
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)

BASE=Path(__file__).resolve().parents[1]
FIG=BASE/"outputs/figures"
OUT=BASE/"data/processed"
FIG.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)

# Load public dataset
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name="target")

# Stratified split preserves class proportions
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.20,stratify=y,random_state=42
)

# Scale inside a pipeline to prevent data leakage
model=Pipeline([
    ("scaler",StandardScaler()),
    ("classifier",LogisticRegression(max_iter=5000,random_state=42))
])

model.fit(X_train,y_train)
pred=model.predict(X_test)
prob=model.predict_proba(X_test)[:,1]

# Metrics
accuracy=accuracy_score(y_test,pred)
precision=precision_score(y_test,pred)
recall=recall_score(y_test,pred)
f1=f1_score(y_test,pred)
auc=roc_auc_score(y_test,prob)
cm=confusion_matrix(y_test,pred)

print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)
print("ROC-AUC:",auc)
print("Confusion matrix:\n",cm)

# Confusion matrix
fig,ax=plt.subplots(figsize=(7,6))
im=ax.imshow(cm)
ax.set_xticks([0,1],data.target_names)
ax.set_yticks([0,1],data.target_names)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix — Logistic Regression")
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16)
plt.colorbar(im,label="Count")
plt.tight_layout()
plt.savefig(FIG/"01_confusion_matrix.png",dpi=220)
plt.close()

# ROC curve
fpr,tpr,_=roc_curve(y_test,prob)
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(fpr,tpr,linewidth=2,label=f"Logistic Regression (AUC={auc:.3f})")
ax.plot([0,1],[0,1],linestyle="--",label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Logistic Regression")
ax.legend()
plt.tight_layout()
plt.savefig(FIG/"02_roc_curve.png",dpi=220)
plt.close()

# Metrics
metric_dict={"Accuracy":accuracy,"Precision":precision,"Recall":recall,
             "F1":f1,"ROC-AUC":auc}
fig,ax=plt.subplots(figsize=(9,6))
bars=ax.bar(metric_dict.keys(),metric_dict.values())
ax.set_ylim(0,1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Summary")
for b,v in zip(bars,metric_dict.values()):
    ax.text(b.get_x()+b.get_width()/2,v+.025,f"{v:.3f}",ha="center")
plt.tight_layout()
plt.savefig(FIG/"03_metric_comparison.png",dpi=220)
plt.close()

# Feature coefficients
coef=pd.Series(model.named_steps["classifier"].coef_[0],index=X.columns)
top=coef.abs().sort_values(ascending=False).head(12).sort_values()
fig,ax=plt.subplots(figsize=(9,7))
ax.barh(top.index,coef[top.index])
ax.axvline(0,linestyle="--")
ax.set_xlabel("Standardized coefficient")
ax.set_title("Most Influential Standardized Features")
plt.tight_layout()
plt.savefig(FIG/"04_feature_coefficients.png",dpi=220)
plt.close()

pd.DataFrame({
    "actual":y_test.values,
    "predicted":pred,
    "probability_class_1":prob
}).to_csv(OUT/"test_predictions.csv",index=False)

pd.DataFrame({
    "metric":list(metric_dict),
    "score":list(metric_dict.values())
}).to_csv(OUT/"model_metrics.csv",index=False)

print("Training and evaluation completed.")
