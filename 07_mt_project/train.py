"""
Mushroom Classification Model Training Script
Trains and saves the final model for production deployment
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(filepath="data/mushroom.csv"):
    """Load and clean the mushroom dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath, sep=";")

    print(f"Initial shape: {df.shape}")

    # Remove duplicates
    duplicates_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {duplicates_before - len(df)} duplicate rows")

    # Drop veil-type (100% single value)
    if "veil-type" in df.columns:
        df = df.drop("veil-type", axis=1)
        print("Dropped 'veil-type' column (no variance)")

    # Handle missing values
    df_clean = df.copy()
    missing_data = df_clean.isnull().sum()
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100

    cols_to_drop = []
    cols_to_impute = []

    for col in df_clean.columns:
        if missing_percent[col] > 0:
            if missing_percent[col] > 80:
                cols_to_drop.append(col)
            else:
                cols_to_impute.append(col)

    # Drop columns with >80% nulls
    if cols_to_drop:
        print(f"Dropping columns with >80% nulls: {cols_to_drop}")
        df_clean = df_clean.drop(cols_to_drop, axis=1)

    # Impute remaining nulls
    for col in cols_to_impute:
        if df_clean[col].dtype == "object":
            df_clean[col].fillna("Unknown", inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    print(f"Final shape after cleaning: {df_clean.shape}")
    print(f"Remaining nulls: {df_clean.isnull().sum().sum()}")

    return df_clean


def prepare_features_and_target(df):
    """Prepare features and target for modeling"""
    print("\nPreparing features...")

    # Separate features and target
    X = df.drop("class", axis=1)
    y = df["class"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print(f"Target classes: {le_target.classes_}")

    return X_encoded, y_encoded, categorical_cols, numerical_cols, label_encoders, le_target


def train_model(X_train, y_train):
    """Train Gradient Boosting model with optimal parameters"""
    print("\nTraining Gradient Boosting Classifier...")

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model


def evaluate_model(model, X_test, y_test, le_target):
    """Evaluate model performance"""
    print("\nEvaluating model on test set...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 70)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    # Feature importance
    feature_importance = model.feature_importances_
    return accuracy, precision, recall, f1, feature_importance


def save_model(model, label_encoders, le_target, filepath="models/model.pkl"):
    """Save trained model and encoders"""
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    model_dict = {
        "model": model,
        "label_encoders": label_encoders,
        "le_target": le_target,
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_dict, f)

    print(f"\n✅ Model saved to {filepath}")


def main():
    """Main training pipeline"""
    print("🍄 MUSHROOM CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # Load and prepare data
    df = load_and_prepare_data("data/mushroom.csv")

    # Prepare features and target
    X, y, cat_cols, num_cols, label_encoders, le_target = prepare_features_and_target(df)

    # Train-test split
    print("\nSplitting data: 80% train, 20% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, precision, recall, f1, feature_importance = evaluate_model(
        model, X_test, y_test, le_target
    )

    # Save model
    save_model(model, label_encoders, le_target, "models/model.pkl")

    print("\n✨ Training pipeline completed successfully!")

    return model


if __name__ == "__main__":
    model = main()
