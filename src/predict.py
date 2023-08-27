from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from data_preprocessing import preprocessed_data 
import pandas as pd
import joblib

def batch_predict():
    # Create columns for non-Ordinal Features
    label_encoder = LabelEncoder()  

    # Load the preprocessed data
    X = preprocessed_data[0]
    y = label_encoder.fit_transform(preprocessed_data[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=500)

    # Load the saved model
    model_filename = 'trained_xgboost_model.joblib'
    gb_model = joblib.load(model_filename)

    gb_predict = gb_model.predict(X_test)
    gb_predict_prob = gb_model.predict_proba(X_test)

    gb_predict_decoded = label_encoder.inverse_transform(gb_predict)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    cat_lbl = ['drunk_driver_involved', 'speeding_driver_involved', 'other']
    print(f"Macro F1 Score: {f1_score(y_test_decoded, gb_predict_decoded, average='macro', labels=cat_lbl)}")
    

if __name__ == "__main__":
    # Perform batch prediction
    predictions = batch_predict()