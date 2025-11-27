"""
Train a Random Forest classifier to distinguish perforators from artifacts.
Uses training data generated from overlap segmentation (generate_training_data.py)
"""
import os
import sys
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config

def load_training_data(data_file):
    """Load training data and extract feature matrix."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    feature_names = ['area', 'perimeter', 'circularity', 'aspect_ratio', 
                     'min_dim', 'max_dim', 'mean_intensity', 'max_intensity',
                     'std_intensity', 'norm_intensity', 'dist_to_edge']
    
    X = []
    y = []
    
    for item in data:
        features = item['features']
        X.append([features.get(fn, 0) for fn in feature_names])
        y.append(item['label'])
    
    return np.array(X), np.array(y), feature_names

def train_classifier(X, y, feature_names):
    """Train classifier and show feature importances."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting (often better than RF for this)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    
    # Cross-validation on training set
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation accuracy: {scores.mean():.2f} (+/- {scores.std()*2:.2f})")
    
    # Fit on training data
    clf.fit(X_train_scaled, y_train)
    
    # Test set evaluation
    y_pred = clf.predict(X_test_scaled)
    print(f"\nTest Set Results:")
    print(classification_report(y_test, y_pred, target_names=['Artifact', 'Perforator']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Neg (correct artifact): {cm[0,0]}")
    print(f"  False Pos (artifact as perf): {cm[0,1]}")
    print(f"  False Neg (perf as artifact): {cm[1,0]}")
    print(f"  True Pos (correct perforator): {cm[1,1]}")
    
    # Retrain on all data for final model
    X_all_scaled = scaler.fit_transform(X)
    clf.fit(X_all_scaled, y)
    
    # Feature importances
    print("\nFeature Importances:")
    importances = list(zip(feature_names, clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = '#' * int(imp * 50)
        print(f"  {name:15s}: {imp:.3f} {bar}")
    
    return clf, scaler

def main():
    data_file = os.path.join(config.OUTPUT_DIR, 'training_data_overlap.json')
    
    if not os.path.exists(data_file):
        print(f"No training data found at {data_file}")
        print("Run generate_training_data.py first!")
        return
    
    X, y, feature_names = load_training_data(data_file)
    
    print(f"Loaded {len(y)} training samples")
    print(f"  Perforators: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Artifacts: {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    if len(y) < 20:
        print("\nNeed at least 20 samples to train reliably.")
        return
    
    clf, scaler = train_classifier(X, y, feature_names)
    
    # Save model
    model_file = os.path.join(config.OUTPUT_DIR, 'perforator_classifier.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'classifier': clf, 'scaler': scaler, 'feature_names': feature_names}, f)
    
    print(f"\nModel saved to {model_file}")
    print("You can now use this in the segmentation pipeline!")

if __name__ == "__main__":
    main()

