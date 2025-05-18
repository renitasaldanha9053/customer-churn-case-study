#model_builder.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tuner import tune_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_and_tune_model(model_name, X_train, y_train):
    if model_name == "logreg":
        model = LogisticRegression(solver='liblinear', random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # Use tuner.py to find best params and return best model
    best_model, cv_results = tune_model(
        model,
        param_grid,
        X_train,
        y_train,
        search_type="grid",   # or "random" if preferred
        cv=5,
        scoring='roc_auc'
    )

    return best_model

# -------- Model selector --------
def get_model(model_name):
    if model_name == "logreg":
        return LogisticRegression(max_iter=10000, solver="lbfgs") 
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run `pip install xgboost`.")
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    else:
        raise ValueError("Model not supported. Choose from: logreg, rf, xgb")

def train_model(X_train, y_train, model_type="logreg"):
    """
    For notebook usage — trains a model and returns it.
    """
    model = get_model(model_type)
    model.fit(X_train, y_train)
    return model

