# train.py Alignment with Notebook Changes

## Summary of Updates

The `train.py` production training script has been updated to align with the final approaches and discoveries from the exploratory notebook. All changes follow the user's requirements:

> "ahora ajustemos el script de train para alinear con estos cambios, principalmente el mantener la columna que no descartamos y con usar los parametros que detectamos como mejores, usaremos solo el algoritmo de gradient bosting"

## Key Changes

### 1. ✅ Feature Encoding: Label → OneHot
**Before:**
- Used `LabelEncoder` for categorical variables
- Problem: Introduces implicit ordinal relationships in categorical data
- Caused unreliability in Logistic Regression

**After:**
- Uses `OneHotEncoder` for categorical variables (13 features → 102 binary features)
- Preserves binary numerical features (4 features)
- Total: 106 final features
- Benefits:
  - Eliminates ordinal bias
  - Works reliably with all model types
  - Matches notebook's successful approach

### 2. ✅ Feature Engineering: spore_print_color_present
**Status:** Implemented in `load_and_prepare_data()`
- Creates binary indicator for spore-print-color presence (89.6% missing)
- Preserves 18.7 percentage point predictive signal
- Included in numerical features (not dropped like other high-missing columns)
- Rationale: Edibility rates differ significantly (27.8% vs 46.5%)

### 3. ✅ Model Selection: Gradient Boosting Only
**Before:**
- Trained 4 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Used cross-validation scoring for model selection
- Stored multiple models unnecessarily

**After:**
- **Only Gradient Boosting** trained and deployed
- Removed model comparison logic
- Removed `cross_val_score` (no longer needed)
- Cleaner production pipeline

### 4. ✅ Hyperparameters: Optimal from GridSearchCV
**Gradient Boosting Optimal Parameters** (5-fold CV, F1 scoring):
```python
learning_rate=0.1      # Controls learning speed
max_depth=7            # Tree depth for regularization
n_estimators=100       # Number of boosting rounds
```

**Previous parameters:**
- learning_rate=0.05 → 0.1 (faster convergence)
- max_depth=5 → 7 (deeper trees for better fit)
- n_estimators=200 → 100 (fewer, more efficient)

## Updated Functions

### `prepare_features_and_target(df)`
**Changes:**
- Replaces all `LabelEncoder` usage with `OneHotEncoder` for categoricals
- Returns: `(X_processed, y_encoded, ohe, le_target, feature_names)`
- Creates proper feature matrix with OHE binary features + numerical columns
- Returns OneHotEncoder object for inference script compatibility

**Output:**
```
Categorical features (13): [cap-shape, cap-surface, cap-color, ...]
Numerical features (4): [cap-diameter, stem-height, stem-width, spore_print_color_present]
OneHotEncoder generated 102 binary features from 13 categorical features
Final feature matrix shape: (60923, 106)
```

### `train_model(X_train, y_train)`
**Changes:**
- Trains only Gradient Boosting with optimal hyperparameters
- Removed cross-validation scoring (not needed at this stage)
- Simplified function - single model instead of 4

**Output:**
```
Training Gradient Boosting Classifier...
Optimal hyperparameters from 5-fold CV with F1 scoring:
✓ Model training completed
  Learning Rate: 0.1
  Max Depth: 7
  Number of Estimators: 100
```

### `save_model(model, ohe, le_target, feature_names, filepath)`
**Changes:**
- Updated to save `OneHotEncoder` instead of individual `label_encoders`
- Added `feature_names` for reference
- Saves complete preprocessing pipeline for inference

**Model Dictionary Contents:**
```python
{
    "model": GradientBoostingClassifier,
    "ohe": OneHotEncoder,              # For categorical encoding
    "le_target": LabelEncoder,          # For target mapping (e/p)
    "feature_names": array of 106 names
}
```

### `main()` Pipeline
**Changes:**
- Updated function signature handling (returns `ohe, le_target, feature_names`)
- Removed unnecessary variables (`cat_cols`, `num_cols`, `label_encoders`)
- Enhanced final output with performance metrics summary

**Pipeline:**
1. `load_and_prepare_data()` → Clean data + create spore_print_color indicator
2. `prepare_features_and_target()` → OneHot encode categoricals
3. `train_test_split()` → 80/20 stratified split
4. `train_model()` → Train GB with optimal hyperparameters
5. `evaluate_model()` → Calculate metrics
6. `save_model()` → Save with preprocessing pipeline

## Test Results

**Training Execution:**
```
🍄 MUSHROOM CLASSIFICATION - TRAINING PIPELINE
======================================================================
Loading data...
- Initial shape: (61069, 21)
- Removed 146 duplicate rows
- Dropped 'veil-type' column (no variance)
- Created 'spore_print_color_present' indicator variable
- Final shape: (60923, 18)

Preparing features with OneHotEncoding...
- Categorical features (13): 102 binary features generated
- Numerical features (4): [cap-diameter, stem-height, stem-width, spore_print_color_present]
- Final feature matrix shape: (60923, 106)

Training Gradient Boosting Classifier...
- Learning Rate: 0.1
- Max Depth: 7
- Number of Estimators: 100

Test Set Performance:
- Accuracy:  0.9997
- Precision: 0.9996
- Recall:    0.9999
- F1-Score:  0.9997
- Confusion Matrix: [[5433, 3], [1, 6748]]
```

**Model Saved:**
```
✅ Model saved to models/model.pkl
   - Model: Gradient Boosting Classifier
   - OneHotEncoder: For categorical feature encoding
   - Target Encoder: ['e' 'p']
   - Feature count: 106
```

## Inference Pipeline Update

The `predict.py` script has been updated to match the new preprocessing:
- Loads OneHotEncoder from saved model
- Applies same preprocessing pipeline (OneHot encoding)
- No longer uses individual LabelEncoders for categories
- Maintains API compatibility

## Algorithm Decision Rationale

### Why OneHotEncoder over LabelEncoder?
1. **Eliminates Ordinal Bias:** Categorical values shouldn't have implicit ordering
2. **Tree Models:** Random Forest & GB work well with binary features
3. **Linear Models:** Logistic Regression now reliable with OHE
4. **Consistency:** All model types can use same preprocessing

### Why Gradient Boosting?
1. **Performance:** Best F1-score in GridSearchCV (0.9995)
2. **Robustness:** Handles complex feature interactions
3. **Efficiency:** Converges with fewer estimators
4. **Regularization:** Built-in via learning_rate and max_depth

### Why Keep spore_print_color_present?
1. **Predictive Power:** 18.7pp difference in edibility rates
2. **Strategic:** Captures signal from 89.6% missing column
3. **Non-Redundant:** Not correlated with other features
4. **Production Value:** Improves model reliability

## Migration from Old train.py

**Breaking Changes:**
- Function signatures changed: `prepare_features_and_target()` now returns different tuple
- Model dict keys changed: `label_encoders` → `ohe`
- Expects `feature_names` in saved model

**Backward Compatibility:**
- Old models incompatible with new inference script
- Retrain required: `python train.py`
- Creates fresh `models/model.pkl` with new format

## Verification

✅ train.py executes end-to-end successfully
✅ Model achieves 0.9997 F1-score on test set
✅ Model dictionary includes all inference components
✅ Prediction test successful (73.47% confidence on sample)
✅ predict.py compatible with new model format

