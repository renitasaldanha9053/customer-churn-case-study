#tuner.py
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def tune_model(model, param_grid, X_train, y_train, 
               search_type="grid", cv=5, n_iter=10, scoring="roc_auc", verbose=1, random_state=42):
    """
    Hyperparameter tuning wrapper.

    Args:
        model: sklearn-compatible estimator (e.g., LogisticRegression, RandomForestClassifier, XGBClassifier)
        param_grid: dict of hyperparameters to search over
        X_train: Training features
        y_train: Training labels
        search_type: "grid" or "random"
        cv: number of cross-validation folds
        n_iter: number of parameter settings sampled for randomized search
        scoring: metric to optimize
        verbose: verbosity level
        random_state: random seed for reproducibility (used in randomized search)

    Returns:
        best_model: model with best found parameters
        cv_results: full cv results object
    """

    logging.info(f"Starting {search_type} search with scoring='{scoring}'...")

    if search_type == "grid":
        searcher = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=-1)
    elif search_type == "random":
        searcher = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, 
                                      cv=cv, scoring=scoring, verbose=verbose, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    searcher.fit(X_train, y_train)
    logging.info(f"Best params: {searcher.best_params_}")
    logging.info(f"Best {scoring}: {searcher.best_score_:.4f}")

    return searcher.best_estimator_, searcher.cv_results_
