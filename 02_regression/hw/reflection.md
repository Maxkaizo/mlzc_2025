# Reflection: Handling Missing Values and the Validation Split (Question 3)

While working on **Question 3**, I ran into an ambiguity: the prompt says to handle missing values either by filling with 0 or with the **mean computed from the training set only**, and then to evaluate on the validation set.
However, it doesn’t explicitly say how to treat the **validation split** under each option, which led me to test both approaches and compare RMSE.

---

## Experiments

### 1) Filling training with 0’s

| Validation handling         | Train RMSE | Val RMSE |
| --------------------------- | ---------- | -------- |
| Filling validation with 0’s | 0.5203     | 0.5174   |
| Without filling validation  | 0.5203     | 0.4969   |

### 2) Filling training with mean (mean computed on train only)

| Validation handling         | Train RMSE | Val RMSE |
| --------------------------- | ---------- | -------- |
| Filling validation with 0’s | 0.4624     | 0.6117   |
| Without filling validation  | 0.4624     | 0.4536   |

---

## What I found

* The best score appeared when I **imputed the training set with the mean** and **left the validation set untouched** (`val RMSE ≈ 0.4536`).
* Investigating further, I confirmed there were **136 NaNs** in the validation predictions (stemming from NaNs in `horsepower`).
* The RMSE function used `pandas.Series.mean()`, which **silently ignores NaNs**. Thus, the metric was computed over a **smaller, cleaner subset** of validation rows—explaining the lower RMSE.

```python
# Example note from my notebook
def rmse(y, y_pred):
    error = y_pred - y
    # pandas' .mean() ignores NaNs by default (skipna=True)
    return np.sqrt((error ** 2).mean())

# A stricter version for transparency:
def rmse_strict(y, y_pred):
    error = y_pred - y
    mask = ~error.isna()
    print(f"RMSE computed on {mask.sum()} of {len(error)} samples")
    return float(np.sqrt(((error[mask]) ** 2).mean()))
```

---

## Interpretation

* For this **homework** question, the intended setup is:
  compute the mean on **train only** and **leave validation unchanged**, then compare RMSEs. That tests robustness when unseen data contains missing values.
* In a **production ML** setting, this would be different: fit the imputer on **train** and **apply the same parameters** to validation and test; compute metrics on the **full** validation set (no silent dropping of NaNs).

---

## Recommendations going forward

1. **Split first**, then perform cleaning, imputation, scaling, and feature engineering—fit all transformations on **train** only.
2. **Apply the same fitted transformations** to validation and test to avoid data leakage.
3. Be explicit about how metrics handle NaNs. If NaNs are ignored, **report how many samples** were actually used in the metric.
4. When exercises intentionally leave validation untouched, **document** this choice and its implications on evaluation.
