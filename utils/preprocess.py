from numpy.typing import ArrayLike
from pandas import DataFrame
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --------------------------------
#  Encode labels (for training only)
# --------------------------------

def encode_labels(y_series) -> tuple[LabelEncoder, ArrayLike]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    return label_encoder, y_encoded

def heart_disease_preprocessor(df: DataFrame, columns_to_impute: list = []) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(missing_values=0 ,strategy="most_frequent")),
        ("scaler", StandardScaler())
    ])

    numeric_cols = columns_to_impute
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols)
    ])

    return preprocessor
