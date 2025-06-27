
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_pipeline(preprocessor) -> Pipeline:
    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42)
    model_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", rf_classifier)
    ])
    return model_pipeline