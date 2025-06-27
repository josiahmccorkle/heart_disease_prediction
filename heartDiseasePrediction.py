import argparse
import os
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils.evaluation import get_pipeline
from utils.io_utils import load_data, output_dir
from utils.preprocess import encode_labels, heart_disease_preprocessor


parser = argparse.ArgumentParser(description="passing in test flag to run predictions")
parser.add_argument("--test", action="store_true", help="Load and run predictions on test.csv instead of training data")
args = parser.parse_args()

output_dir = "./outputs"
model_path = os.path.join(output_dir, "model.pkl")

# --------------------------------
#  Load Data
# --------------------------------
df = load_data()
encoder_path = os.path.join(output_dir, "encoder.pkl")
X = df.drop(columns=["target"])

if not args.test:
    label_encoder, y = encode_labels(df["target"])
    columns_to_impute = ["slope", "thal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = heart_disease_preprocessor(X, columns_to_impute)
    pipeline = get_pipeline(preprocessor)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)
else:
    label_encoder = joblib.load(encoder_path)
    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(X)
    y_true = df["target"]
    print(classification_report(y_true, predictions))


if not os.path.exists(output_dir):
    os.makedirs(output_dir)



