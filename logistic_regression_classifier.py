# Logistic Regression for Breast Cancer Classification
# Task 4: Binary Classification
# Dataset: Breast Cancer Wisconsin Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def load_dataset():
    """Load and return the breast cancer dataset"""
    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"Classes: {np.unique(y)} (0=Malignant, 1=Benign)")
    
    return X, y, data.feature_names

def explore_data(X, y):
    """Explore the dataset characteristics"""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Basic info
    total_samples = len(y)
    malignant_count = sum(y == 0)
    benign_count = sum(y == 1)
    
    print(f"Total samples: {total_samples}")
    print(f"Malignant cases: {malignant_count} ({malignant_count/total_samples*100:.1f}%)")
    print(f"Benign cases: {benign_count} ({benign_count/total_samples*100:.1f}%)")
    
    # Create visualization of class distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Class distribution pie chart
    axes[0,0].pie([malignant_count, benign_count], 
                  labels=['Malignant', 'Benign'], 
                  autopct='%1.1f%%', 
                  colors=['lightcoral', 'lightblue'])
    axes[0,0].set_title('Class Distribution')
    
    # Feature correlation heatmap (top 10 features)
    correlation_matrix = X.iloc[:, :10].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0,1], cbar_kws={'shrink': .8})
    axes[0,1].set_title('Feature Correlations (First 10 Features)')
    
    # Box plot for some key features
    key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    data_for_box = []
    labels_for_box = []
    
    for feature in key_features:
        data_for_box.extend([X[feature][y==0].values, X[feature][y==1].values])
        labels_for_box.extend([f'{feature}\n(Malignant)', f'{feature}\n(Benign)'])
    
    axes[1,0].boxplot(data_for_box, labels=labels_for_box)
    axes[1,0].set_title('Key Features Distribution by Class')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Feature statistics
    mean_features = [col for col in X.columns if 'mean' in col][:8]
    malignant_means = X[y==0][mean_features].mean()
    benign_means = X[y==1][mean_features].mean()
    
    x_pos = np.arange(len(mean_features))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, malignant_means, width, label='Malignant', color='lightcoral')
    axes[1,1].bar(x_pos + width/2, benign_means, width, label='Benign', color='lightblue')
    axes[1,1].set_xlabel('Features')
    axes[1,1].set_ylabel('Mean Values')
    axes[1,1].set_title('Mean Feature Values by Class')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([f.replace('mean ', '') for f in mean_features], rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X, y

def preprocess_data(X, y):
    """Split and scale the data"""
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Feature scaling completed using StandardScaler")
    
    # Visualize scaling effect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before scaling
    ax1.hist(X_train.iloc[:, 0], bins=30, alpha=0.7, color='blue', label='Before scaling')
    ax1.set_title('Feature Distribution Before Scaling')
    ax1.set_xlabel('Feature Values')
    ax1.set_ylabel('Frequency')
    
    # After scaling
    ax2.hist(X_train_scaled[:, 0], bins=30, alpha=0.7, color='green', label='After scaling')
    ax2.set_title('Feature Distribution After Scaling')
    ax2.set_xlabel('Standardized Values')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train):
    """Train the logistic regression model"""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Initialize and train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Logistic Regression model trained successfully")
    print(f"Convergence achieved in {model.n_iter_[0]} iterations")
    
    return model

def visualize_sigmoid():
    """Create sigmoid function visualization"""
    print("\n" + "="*50) 
    print("SIGMOID FUNCTION")
    print("="*50)
    
    # Generate sigmoid curve
    z = np.linspace(-10, 10, 100)
    sigmoid_values = 1 / (1 + np.exp(-z))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sigmoid curve
    ax1.plot(z, sigmoid_values, 'b-', linewidth=3, label='σ(z) = 1/(1 + e^(-z))')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='Decision boundary (0.5)')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.6)
    ax1.set_xlabel('z (linear combination)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Sigmoid Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sigmoid properties demonstration
    key_points_z = [-2, -1, 0, 1, 2]
    key_points_sigmoid = [1/(1 + np.exp(-z)) for z in key_points_z]
    
    ax2.scatter(key_points_z, key_points_sigmoid, color='red', s=100, zorder=5)
    ax2.plot(z, sigmoid_values, 'b-', linewidth=2, alpha=0.7)
    
    for i, (z_val, sig_val) in enumerate(zip(key_points_z, key_points_sigmoid)):
        ax2.annotate(f'({z_val}, {sig_val:.3f})', 
                    (z_val, sig_val), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    ax2.set_xlabel('z values')
    ax2.set_ylabel('σ(z) values') 
    ax2.set_title('Key Points on Sigmoid Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigmoid_function.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    return y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred):
    """Create and visualize confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add metric calculations
    tn, fp, fn, tp = cm.ravel()
    plt.figtext(0.02, 0.1, f'TN: {tn}  FP: {fp}\nFN: {fn}  TP: {tp}', 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_roc_and_metrics(y_test, y_pred_proba):
    """Plot ROC curve and other performance metrics"""
    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    axes[0,0].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random classifier')
    axes[0,0].set_xlim([0.0, 1.0])
    axes[0,0].set_ylim([0.0, 1.05])
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend(loc="lower right")
    axes[0,0].grid(True, alpha=0.3)
    
    # Probability distribution
    malignant_probs = y_pred_proba[y_test == 0]
    benign_probs = y_pred_proba[y_test == 1]
    
    axes[0,1].hist(malignant_probs, bins=20, alpha=0.7, color='red', 
                   label='Malignant', density=True)
    axes[0,1].hist(benign_probs, bins=20, alpha=0.7, color='blue', 
                   label='Benign', density=True)
    axes[0,1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    axes[0,1].set_xlabel('Predicted Probability')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Probability Distribution by Class')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Threshold analysis
    thresholds_analysis = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds_analysis:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred_thresh))
        recalls.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))
    
    axes[1,0].plot(thresholds_analysis, precisions, 'o-', label='Precision', color='blue')
    axes[1,0].plot(thresholds_analysis, recalls, 's-', label='Recall', color='red')
    axes[1,0].plot(thresholds_analysis, f1_scores, '^-', label='F1-Score', color='green')
    axes[1,0].set_xlabel('Threshold')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Metrics vs Threshold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Classification report visualization
    from sklearn.metrics import classification_report
    report = classification_report(y_test, (y_pred_proba >= 0.5).astype(int), output_dict=True)
    
    classes = ['Malignant', 'Benign']
    metrics = ['precision', 'recall', 'f1-score']
    
    precision_vals = [report['0']['precision'], report['1']['precision']]
    recall_vals = [report['0']['recall'], report['1']['recall']]
    f1_vals = [report['0']['f1-score'], report['1']['f1-score']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    axes[1,1].bar(x - width, precision_vals, width, label='Precision', color='lightblue')
    axes[1,1].bar(x, recall_vals, width, label='Recall', color='lightcoral')
    axes[1,1].bar(x + width, f1_vals, width, label='F1-Score', color='lightgreen')
    
    axes[1,1].set_xlabel('Classes')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Performance by Class')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(classes)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance"""
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Get coefficients
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['coefficient']:.4f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top 15 features by absolute coefficient
    top_features = feature_importance.head(15)
    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
    
    ax1.barh(range(len(top_features)), top_features['coefficient'], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels([name[:25] for name in top_features['feature']])
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title('Top 15 Feature Coefficients\n(Red: Malignant indicators, Blue: Benign indicators)')
    ax1.grid(True, alpha=0.3)
    
    # Feature categories analysis
    mean_features = [f for f in feature_names if 'mean' in f]
    worst_features = [f for f in feature_names if 'worst' in f]
    se_features = [f for f in feature_names if 'error' in f]
    
    mean_importance = np.mean([abs(model.coef_[0][list(feature_names).index(f)]) for f in mean_features])
    worst_importance = np.mean([abs(model.coef_[0][list(feature_names).index(f)]) for f in worst_features])
    se_importance = np.mean([abs(model.coef_[0][list(feature_names).index(f)]) for f in se_features])
    
    categories = ['Mean Features', 'Worst Features', 'Standard Error Features']
    importance_values = [mean_importance, worst_importance, se_importance]
    
    ax2.bar(categories, importance_values, color=['lightblue', 'lightcoral', 'lightgreen'])
    ax2.set_ylabel('Average |Coefficient|')
    ax2.set_title('Feature Category Importance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete analysis"""
    print("🔬 BREAST CANCER CLASSIFICATION USING LOGISTIC REGRESSION")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_dataset()
    
    # Explore data
    X, y = explore_data(X, y)
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model
    model = train_logistic_regression(X_train_scaled, y_train)
    
    # Visualize sigmoid function
    visualize_sigmoid()
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test)
    
    # Create visualizations
    cm = plot_confusion_matrix(y_test, y_pred)
    plot_roc_and_metrics(y_test, y_pred_proba)
    analyze_feature_importance(model, feature_names)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📊 Generated visualizations:")
    print("   • data_exploration.png")
    print("   • feature_scaling.png") 
    print("   • sigmoid_function.png")
    print("   • confusion_matrix.png")
    print("   • performance_analysis.png")
    print("   • feature_importance.png")
    print("=" * 60)

if __name__ == "__main__":
    main()