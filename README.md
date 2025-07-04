# heart_disease_prediction

Heart Disease Prediction - Predict if a patient has heart disease based on features like age, cholesterol, etc.

## Features
- Preprocessing (imputation, scaling)
- Label encoding
- Model training and evaluation
- Test-mode for reloading and prediction
- Modular, reusable code structure
- Outputs trained model and encoder as `.pkl` files

## Usage

### Im using pyenv to maintain my current version of `Python 3.13.2`


to train the model:
```bash
python heartDiseasePrediction.py
```
to test the model:
```bash
python heartDiseasePrediction.py --test
```


### Notes
- The main script is heartDiseasePrediction.py.
- Preprocessing includes imputation for missing values and scaling of features.
- Label encoding is used for the target variable.
- The model and encoder are saved in the outputs/ directory after training.
- In test mode, the script loads the saved model and encoder to make predictions and prints a classification report.
