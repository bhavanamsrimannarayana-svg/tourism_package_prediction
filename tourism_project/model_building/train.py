# for data manipulation
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

base_url = "https://huggingface.co/datasets/srbhavan/tourism-package-prediction/resolve/main/"

Xtrain = pd.read_csv(base_url + "Xtrain.csv")
Xtest = pd.read_csv(base_url + "Xtest.csv")
ytrain = pd.read_csv(base_url + "ytrain.csv")
ytest = pd.read_csv(base_url + "ytest.csv")

# Grouping 1: Nominal Categories (No implicit hierarchy -> One-Hot Encode)
nominal_cols = ['Gender', 'MaritalStatus', 'TypeofContact', 'Occupation']

# Grouping 2: Ordinal Categories (Clear structural progression -> Ordinal Mapping)
ordinal_cols = ['Designation', 'ProductPitched']

# Explicitly define the rank hierarchy strings to avoid default alphabetical mapping
designation_order = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
product_order = ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King']

# Grouping 3 & 4: Numeric, Continuous, Binary Flags, and CityTier 
# (XGBoost tree-splits natively handle these as integers/floats without scaling)
numeric_and_passthrough_cols = [
    'Age', 'MonthlyIncome', 'DurationOfPitch', 
    'NumberOfTrips', 'NumberOfFollowups', 'PitchSatisfactionScore', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting',
    'Passport', 'OwnCar', 
    'CityTier'  # Stays here to bypass any scaling or encoding overhead
]

# ==========================================
# 3. STEP THREE: THE SKLEARN COMPOSER PIPELINE
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        # 1. Nominal Engine: Ignores missing categories at live production inference
        ('nominal_transformer', 
         OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         nominal_cols),
        
        # 2. Ordinal Engine: Formats ranks matching actual domain reality
        ('ordinal_transformer', 
         OrdinalEncoder(categories=[designation_order, product_order], handle_unknown='use_encoded_value', unknown_value=-1), 
         ordinal_cols),
         
        # 3. Passthrough Engine: Tells the system to explicitly protect these features
        ('passthrough_transformer', 
         'passthrough', 
         numeric_and_passthrough_cols)
    ],
    remainder='drop' # Automatically drops completely unrelated metadata columns like 'CustomerID'
)


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight


# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "srbhavan/tourism-package-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_prediction_model_v1.joblib",
        path_in_repo="best_tourism_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
