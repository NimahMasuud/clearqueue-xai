## Results on the held-out test set

```json
{
  "priority_models": {
    "Logistic regression": {
      "macro_f1": 0.9369358298145664,
      "high_precision": 0.9649122807017544,
      "high_recall": 0.9649122807017544,
      "high_f1": 0.9649122807017544
    },
    "Linear SVM": {
      "macro_f1": 0.9293459004535078,
      "high_precision": 0.9581881533101045,
      "high_recall": 0.9649122807017544,
      "high_f1": 0.9615384615384616
    },
    "Random forest": {
      "macro_f1": 0.8944443510674853,
      "high_precision": 0.9518900343642611,
      "high_recall": 0.9719298245614035,
      "high_f1": 0.9618055555555556
    }
  },
  "priority_errors": {
    "missed_high_rate": 0.03508771929824561,
    "false_high_rate": 0.03508771929824561
  },
  "department_accuracy": 1.0,
  "department_macro_f1": 1.0,
  "stability_under_rewording": 0.9966666666666667,
  "macro_f1_by_length": {
    "shorter": 0.9436429646347895,
    "longer": 0.9246706000523326
  },
  "high_recall_by_department": {
    "Electoral Services Desk": 0.88,
    "Emergency & Public Safety": 1.0,
    "Health Services": 0.9811320754716981,
    "Power Distribution": 0.9565217391304348,
    "Roads & Transport": 0.9459459459459459,
    "Waste Management": 0.8,
    "Water & Sanitation": 0.98
  },
  "explanation_overlap_coef_vs_lime": 0.6619642857142857,
  "share_sent_to_human_review": 0.19,
  "test_size": 600
}
```
