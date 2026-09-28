# Training review notes

The bundled model was trained on the supplied DDoS-specific CICIDS2017 slice and is therefore a binary BENIGN-vs-DDoS classifier.

For the next retraining pass on a larger multi-attack dataset:

1. Combine the CICIDS2017 daily CSVs and preserve the original `Label` values.
2. Remove constant columns and exact duplicate columns before feature selection.
3. Fit mutual-information selection inside each training fold only.
4. Use a grouped/time-aware outer evaluation whenever a session/capture identifier is available.
5. Report macro-F1 and per-class recall for multi-class evaluation, not only accuracy.
6. Report permutation or SHAP-based importance alongside native XGBoost gain so redundant variables are easier to interpret.
