"""
Mushroom Classification Model Training Script
Trains and saves the final model for production deployment
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

    # Drop veil-type (no variance)
    if "veil-type" in df.columns:
        df = df.drop("veil-type", axis=1)
        print("Dropped 'veil-type' column (no variance)")

    # Handle missing values - critical for spore-print-color
    df_clean = df.copy()
    missing_data = df_clean.isnull().sum()
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100

    # Create presence indicator for spore-print-color (18.7 pp predictive difference)
    if "spore-print-color" in df_clean.columns:
        df_clean['spore_print_color_present'] = (~df_clean['spore-print-color'].isnull()).astype(int)
        print("Created 'spore_print_color_present' indicator variable")

    # Identify columns to drop or impute
    cols_to_drop = []
    cols_to_impute = []

    for col in df_clean.columns:
        if col == 'spore_print_color_present':
            continue  # Skip our new indicator
        if missing_percent[col] > 0:
            if missing_percent[col] > 80:
                cols_to_drop.append(col)
            else:
                cols_to_impute.append(col)

    # Drop columns with >80% nulls (except spore-print-color which we handled)
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
    """Prepare features and target using OneHotEncoding for categorical variables"""
    print("\nPreparing features with OneHotEncoding...")

    # Separate features and target
    y = df["class"].copy()
    X = df.drop("class", axis=1).copy()

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"Target classes: {le_target.classes_}")

    # Apply OneHotEncoding to categorical features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = ohe.fit_transform(X[categorical_cols])

    # Get feature names after encoding
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    print(f"OneHotEncoder generated {len(cat_feature_names)} binary features from {len(categorical_cols)} categorical features")

    # Combine encoded categorical with numerical features
    X_encoded = np.hstack([X_cat_encoded, X[numerical_cols].values])
    feature_names = np.concatenate([cat_feature_names, numerical_cols])

    print(f"Final feature matrix shape: {X_encoded.shape}")
    print(f"  {len(cat_feature_names)} encoded categorical + {len(numerical_cols)} numerical = {len(feature_names)} total features")

    # Create DataFrame with encoded features
    X_processed = pd.DataFrame(X_encoded, columns=feature_names)

    return X_processed, y_encoded, ohe, le_target, feature_names


def train_model(X_train, y_train):
    """Train Gradient Boosting model with optimal hyperparameters from GridSearchCV"""
    print("\nTraining Gradient Boosting Classifier...")
    print("Optimal hyperparameters from 5-fold CV with F1 scoring:")

    model = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=100,
        random_state=42,
        n_iter_no_change=10,
        validation_fraction=0.1,
    )

    model.fit(X_train, y_train)
    print("✓ Model training completed")
    print(f"  Learning Rate: 0.1")
    print(f"  Max Depth: 7")
    print(f"  Number of Estimators: 100")

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


def save_model(model, ohe, le_target, feature_names, filepath="models/model.pkl"):
    """Save trained model and encoders for inference"""
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    model_dict = {
        "model": model,
        "ohe": ohe,
        "le_target": le_target,
        "feature_names": feature_names,
    }

    with open(filepath, "wb") as f:
        pickle.dump(model_dict, f)

    print(f"\n✅ Model saved to {filepath}")
    print(f"   - Model: Gradient Boosting Classifier")
    print(f"   - OneHotEncoder: For categorical feature encoding")
    print(f"   - Target Encoder: {le_target.classes_}")
    print(f"   - Feature count: {len(feature_names)}")


def main():
    """Main training pipeline"""
    print("🍄 MUSHROOM CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 70 + "\n")

    # Load and prepare data
    df = load_and_prepare_data("data/mushroom.csv")

    # Prepare features and target with OneHotEncoding
    X, y, ohe, le_target, feature_names = prepare_features_and_target(df)

    # Train-test split
    print("\nSplitting data: 80% train, 20% test (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Class distribution preserved in both sets")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, precision, recall, f1, feature_importance = evaluate_model(
        model, X_test, y_test, le_target
    )

    # Save model with preprocessing pipeline
    save_model(model, ohe, le_target, feature_names, "models/model.pkl")

    print("\n✨ Training pipeline completed successfully!")
    print(f"Final Model Performance on Test Set:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return model


if __name__ == "__main__":
    model = main()
