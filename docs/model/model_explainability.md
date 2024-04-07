# Model Explainability

## SHAP

- SHAP (SHapley Additive exPlanations), originally created in 2017, has been designed, using the game theory, to identify what is the marginal contribution of each feature within the model.

```Python
import shap
# load JS visualization code to notebook
shap.initjs()
```

- Create SHAP Explainer & SHAP values based on the training set

```Python
# Overall calculation of the SHAP model and values
shap_explainer = shap.TreeExplainer(clf_xgb) # XGBoost model -> use TreeExplainer
shap_values = shap_explainer.shap_values(X_train_pre)
```

### SHAP Local Explainability

- Explain the SHAP for a single prediction
  - The above explanation shows features each contributing to push the **model output f(x)** from the **base value** (average contribution of all parameters within the model).
  - Features pushing the prediction higher are shown in Red
  - Features pushing the prediction lower are in Blue
  - The longer the length of each feature, the more impact on the model output

```Python
index_choice = 0
shap.force_plot(shap_explainer.expected_value, shap_values[index_choice, :], X_train_pre.iloc[index_choice, :])
```

### SHAP Global Explainability

```Python
# Summary plot
shap.summary_plot(shap_values, X_train_pre, feature_names=X_train_pre.columns, plot_type="bar")
plt.show()
```
