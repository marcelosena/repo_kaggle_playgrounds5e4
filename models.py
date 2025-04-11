import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error # Example metric, can be changed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Example preprocessor

# Added imports for PyTorch FNN
import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping, ProgressBar

# Added import for XGBoost
import xgboost as xgb

# Added imports for LightGBM and CatBoost
import lightgbm as lgb
import catboost as cb

# Imports for cuML KNN (conditional)
try:
    import cudf
    from cuml.neighbors import KNeighborsRegressor as CumlKNeighborsRegressor
    from cuml.preprocessing import StandardScaler as CumlStandardScaler
    from cuml.metrics.regression import mean_squared_error as cuml_mean_squared_error
    import rmm # Check if GPU memory pool can be initialized
    rmm.reinitialize(pool_allocator=True)
    # Optional: Check for actual GPU device presence if needed
    # import cupy
    # if cupy.cuda.runtime.getDeviceCount() == 0:
    #    raise ImportError("No CUDA devices found")
    CUML_AVAILABLE = True
    print("\ncuDF/cuML found and GPU accessible. Will use cuML for KNN.")
except ImportError:
    CUML_AVAILABLE = False
    print("\nWarning: cuDF/cuML not found or GPU unavailable. Will use scikit-learn for KNN.")
except Exception as e:
    # Catch other potential errors during RMM init or device check
    CUML_AVAILABLE = False
    print(f"\nWarning: Error initializing cuML ({e}). Will use scikit-learn for KNN.")

# Define scoring metric globally or pass as argument if varies
SCORING = 'neg_root_mean_squared_error' # Example, choose appropriate metric

# Import sklearn KNN and explicitly alias StandardScaler
from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# Added import for skorch train_split
import skorch

# --- Master Parameters ---
# Define RANDOM_STATE here as well for use within model functions
RANDOM_STATE = 42

K_FOLDS = 3
# RANDOM_STATE = 42 # Already defined above
DATA_PATH = "train.csv"
TARGET_COLUMN = "Listening_Time_minutes"

# --- Model Definition Functions ---
class PyTorchFNN(nn.Module):
    """Simple PyTorch Feedforward Neural Network for regression with Dropout."""
    def __init__(self, n_features_in, hidden_units=64, activation_fn=nn.ReLU(), dropout_rate=0.2):
        super().__init__()
        self.layer_1 = nn.Linear(n_features_in, hidden_units)
        self.activation = activation_fn
        self.dropout_1 = nn.Dropout(dropout_rate) # Dropout layer 1
        self.layer_2 = nn.Linear(hidden_units, hidden_units // 2)
        # No activation needed right before output usually, but add dropout
        self.dropout_2 = nn.Dropout(dropout_rate) # Dropout layer 2
        self.output_layer = nn.Linear(hidden_units // 2, 1)

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.dropout_1(x) # Apply dropout after first activation
        x = self.activation(self.layer_2(x))
        x = self.dropout_2(x) # Apply dropout after second activation
        x = self.output_layer(x)
        return x

# --- Training and Evaluation Functions ---

def train_evaluate_gbt(X_train, y_train, X_test, y_test):
    """Trains, tunes (using FAST RandomizedSearch), and evaluates a Gradient Boosting Regressor."""
    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()),
        ('gbt', GradientBoostingRegressor(random_state=42))
    ])
    # Define hyperparameter distributions for FAST RandomizedSearch
    # Use minimal values for speed
    param_dist = {
        'gbt__n_estimators': [10], # Drastically reduced number of trees
        'gbt__learning_rate': [0.1], # Fixed learning rate
        'gbt__max_depth': [2] # Drastically reduced depth
    }

    # Use RandomizedSearchCV with minimal iterations
    n_iter_search = 1 # Minimal number of iterations
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=3, # Keep 3 folds for some robustness
        scoring=SCORING,
        n_jobs=-1, # Keep parallel CV
        random_state=42 # for reproducibility of sampling
    )

    print("\nRunning FAST Gradient Boosting tuning...")
    random_search.fit(X_train, y_train)
    # Since n_iter=1 and only one option per param, the "best" params are fixed
    print(f"GBT parameters used (Fast): {random_search.best_params_}")
    best_model = random_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))
    return {
        'model': best_model,
        'best_params': random_search.best_params_,
        'cv_score': random_search.best_score_,
        'test_score': test_score,
        'backend': 'Sklearn'
    }

def train_evaluate_ridge(X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates a Ridge Regression model."""
    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()),
        ('ridge', Ridge(random_state=42))
    ])
    param_grid = {
        'ridge__alpha': np.logspace(-3, 3, 7)
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=SCORING, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Ridge parameters found: {grid_search.best_params_}")
    print(f"Best Ridge cross-validation score ({SCORING}): {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))
    return {
        'model': best_model,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'test_score': test_score,
        'backend': 'Sklearn'
    }

def train_evaluate_fnn(X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates a PyTorch Feedforward Neural Network using skorch."""
    # Ensure data is float32 for PyTorch
    # y_train needs to be numpy before reshape
    y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_train_torch = y_train_np.astype(np.float32).reshape(-1, 1) # Target needs reshaping

    # Skorch wrapper for PyTorch model
    net = NeuralNetRegressor(
        module=PyTorchFNN,
        module__n_features_in=X_train.shape[1], # Pass input features dynamically
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=0.01, # Default LR, can be tuned
        max_epochs=10, # Limit to 10 epochs (as requested earlier)
        batch_size=32, # Default batch size, can be tuned
        # Use skorch's internal train_split for validation during hyperparameter search / early stopping
        train_split=skorch.dataset.ValidSplit(cv=0.15, stratified=False), # Use 15% of train data for internal validation
        callbacks=[
            # Early stopping now monitors 'valid_loss' from the internal split
            ('early_stopping', EarlyStopping(monitor='valid_loss', patience=5, threshold=1e-4, lower_is_better=True)),
            ('progress_bar', ProgressBar())
        ],
        verbose=0, # Keep verbose=0 as ProgressBar handles output
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()), # Sklearn scaler works fine before skorch
        ('net', net)
    ])

    # Define hyperparameter grid for Skorch/PyTorch
    param_grid = {
        'net__lr': [0.01, 0.001, 1e-4],
        'net__max_epochs': [1000],
        'net__module__hidden_units': [1000, 2000],
        'net__module__dropout_rate': [0.1, 0.3] # Added dropout rate to grid search
    }

    # GridSearchCV uses its own CV splits on the provided training data.
    # Skorch's train_split is used *within* each fit call inside GridSearchCV for early stopping.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=SCORING, n_jobs=-1, refit=True)

    # Fit the pipeline (scaler + skorch net)
    # Cast X_train here as pipeline expects numpy
    grid_search.fit(X_train.astype(np.float32), y_train_torch)

    print(f"Best FNN parameters found: {grid_search.best_params_}")
    best_cv_score = grid_search.best_score_
    if SCORING.startswith('neg_'):
        best_cv_score = -best_cv_score
    print(f"Best FNN cross-validation score ({SCORING.replace('neg_','')+' (positive)'}): {best_cv_score:.4f}")

    best_model_pipeline = grid_search.best_estimator_
    # Ensure X_test is also float32 for prediction
    y_pred_test = best_model_pipeline.predict(X_test.astype(np.float32))
    # Ensure y_test is numpy array for scoring
    y_test_np = y_test if isinstance(y_test, np.ndarray) else y_test.values
    test_score = np.sqrt(mean_squared_error(y_test_np, y_pred_test))

    best_params_cleaned = {k.replace('net__', '').replace('module__', ''): v for k, v in grid_search.best_params_.items()}

    return {
        'model': best_model_pipeline,
        'best_params': best_params_cleaned,
        'cv_score': best_cv_score,
        'test_score': test_score,
        'backend': 'PyTorch/Skorch'
    }

def train_evaluate_xgb(X_train, y_train, X_test, y_test):
    """Trains, tunes, and evaluates an XGBoost Regressor."""
    # Removed X_val, y_val arguments
    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()),
        ('xgb', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1))
        # Use default n_jobs=-1 to utilize all cores for XGBoost itself
    ])

    # Define hyperparameter grid for XGBoost
    # Reduced grid for faster example execution
    param_grid = {
        'xgb__n_estimators': [200, 400, 800], # [100, 200, 300]
        'xgb__learning_rate': [0.05, 0.1, 0.3], # [0.01, 0.1, 0.3]
        'xgb__max_depth': [9, 11, 13], # [3, 5, 7]
        # 'xgb__subsample': [0.7, 1.0],
        # 'xgb__colsample_bytree': [0.7, 1.0]
    }

    # Perform grid search
    # Note: GridSearchCV n_jobs controls parallelism for CV folds/param combinations.
    # XGBoost's n_jobs controls its internal parallelism.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=SCORING, n_jobs=1) # Use n_jobs=1 for GridSearchCV if XGBoost uses n_jobs=-1 to avoid oversubscription
    # Alternatively, set xgb(n_jobs=1) and GridSearchCV(n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(f"Best XGBoost parameters found: {grid_search.best_params_}")
    best_cv_score = grid_search.best_score_
    if SCORING.startswith('neg_'):
        best_cv_score = -best_cv_score
    print(f"Best XGBoost cross-validation score ({SCORING.replace('neg_','')+' (positive)'}): {best_cv_score:.4f}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Clean up param names for reporting
    best_params_cleaned = {k.replace('xgb__', ''): v for k, v in grid_search.best_params_.items()}

    return {
        'model': best_model,
        'best_params': best_params_cleaned,
        'cv_score': best_cv_score,
        'test_score': test_score,
        'backend': 'XGBoost'
    }

def train_evaluate_knn(X_train, y_train, X_test, y_test):
    """Trains and evaluates K-Nearest Neighbors Regressor.

    Uses cuML (GPU) if available and working, otherwise falls back to
    scikit-learn (CPU).
    Uses fixed hyperparameters for simplicity (no GridSearchCV/RandomizedSearchCV).
    """
    # Removed X_val, y_val arguments
    n_neighbors = 10 # Fixed hyperparameter for both backends
    result_model = None
    test_score = np.nan
    backend = 'unknown' # To track which path was taken

    if CUML_AVAILABLE:
        # --- Try cuML (GPU) Backend First ---
        print("\nAttempting KNN with cuML backend...")
        backend = 'cuml'
        try:
            # Convert data
            X_train_gpu = cudf.DataFrame.from_pandas(X_train.astype(np.float32))
            y_train_gpu = cudf.Series(y_train.astype(np.float32).values)
            X_test_gpu = cudf.DataFrame.from_pandas(X_test.astype(np.float32))
            y_test_gpu = cudf.Series(y_test.astype(np.float32).values)

            # Scale data
            scaler = CumlStandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_gpu)
            X_test_scaled = scaler.transform(X_test_gpu)

            # Train model
            model = CumlKNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(X_train_scaled, y_train_gpu)

            # Predict and Evaluate
            y_pred_gpu = model.predict(X_test_scaled)
            mse = cuml_mean_squared_error(y_test_gpu, y_pred_gpu)
            test_score = np.sqrt(mse)

            # Store the scaler along with the model for prediction later
            result_model = {'model': model, 'scaler': scaler}
            print(f"KNN ({backend}) Test Score (RMSE): {test_score:.4f}")
            # Return successfully if cuML worked
            return {
                'model': result_model,
                'backend': backend,
                'best_params': {'n_neighbors': n_neighbors},
                'cv_score': np.nan,
                'test_score': test_score
            }

        except Exception as e:
            print(f"Error during cuML KNN execution: {e}. Falling back to scikit-learn.")
            # Do not modify CUML_AVAILABLE globally
            # Proceed to the sklearn block below
            backend = 'sklearn_fallback' # Indicate we are falling back
    
    # --- Use scikit-learn (CPU) Backend ---
    # This block runs if CUML_AVAILABLE was initially False OR if the cuML try block failed
    print(f"\nUsing KNN with scikit-learn backend... (Reason: { 'cuML not available' if not CUML_AVAILABLE else 'cuML failed' })")
    backend = 'sklearn'
    try:
        pipeline = Pipeline([
            ('scaler', SklearnStandardScaler()),
            ('knn', SklearnKNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predict and Evaluate
        y_pred = pipeline.predict(X_test)
        test_score = np.sqrt(mean_squared_error(y_test, y_pred))

        result_model = pipeline # Store the entire pipeline
        print(f"KNN ({backend}) Test Score (RMSE): {test_score:.4f}")
        # Return after successful sklearn run
        return {
            'model': result_model,
            'backend': backend,
            'best_params': {'n_neighbors': n_neighbors},
            'cv_score': np.nan,
            'test_score': test_score
        }

    except Exception as e:
         print(f"Error during scikit-learn KNN execution: {e}. Skipping KNN.")
         # Return error if sklearn fails
         return {
            'model': 'Sklearn Error',
            'backend': backend,
            'best_params': {'n_neighbors': n_neighbors},
            'cv_score': np.nan,
            'test_score': np.nan
        }

def train_evaluate_lgbm(X_train, y_train, X_test, y_test):
    """Trains, tunes (RandomizedSearch), and evaluates a LightGBM Regressor."""
    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()), # Scaling is optional but can sometimes help
        ('lgbm', lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    # Simplified parameter distribution for faster tuning
    param_dist = {
        'lgbm__n_estimators': [50, 100, 150],
        'lgbm__learning_rate': [0.05, 0.1, 0.2],
        'lgbm__num_leaves': [20, 31, 40], # Default is 31
        'lgbm__max_depth': [-1, 5, 10], # -1 means no limit
        # 'lgbm__reg_alpha': [0, 0.1, 0.5],
        # 'lgbm__reg_lambda': [0, 0.1, 0.5]
    }

    n_iter_search = 4 # Number of parameter settings sampled
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=3,
        scoring=SCORING,
        n_jobs=1, # Let LightGBM handle internal parallelism with n_jobs=-1
        random_state=RANDOM_STATE,
        refit=True
    )

    print("Running LightGBM tuning...")
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print(f"Best LightGBM parameters found: {best_params}")
    best_cv_score = random_search.best_score_
    if SCORING.startswith('neg_'):
        best_cv_score = -best_cv_score
    print(f"Best LightGBM cross-validation score ({SCORING.replace('neg_','')+' (positive)'}): {best_cv_score:.4f}")

    best_model = random_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))

    best_params_cleaned = {k.replace('lgbm__', ''): v for k, v in best_params.items()}

    return {
        'model': best_model,
        'best_params': best_params_cleaned,
        'cv_score': best_cv_score,
        'test_score': test_score,
        'backend': 'LightGBM'
    }

def train_evaluate_catboost(X_train, y_train, X_test, y_test):
    """Trains, tunes (RandomizedSearch), and evaluates a CatBoost Regressor."""
    # CatBoost handles categorical features internally, but scaling numeric might still be beneficial
    # Note: If passing categorical features directly, ensure they are NOT label encoded beforehand.
    # Our current preprocessing does label encode, so treat all as numeric for CatBoost here.
    pipeline = Pipeline([
        ('scaler', SklearnStandardScaler()),
        ('cat', cb.CatBoostRegressor(random_state=RANDOM_STATE,
                                       verbose=0, # Suppress verbose training output
                                       allow_writing_files=False)) # Avoid creating temp files
    ])

    # Simplified parameter distribution
    param_dist = {
        'cat__iterations': [50, 100, 150], # Equivalent to n_estimators
        'cat__learning_rate': [0.05, 0.1, 0.2],
        'cat__depth': [4, 6, 8],
        # 'cat__l2_leaf_reg': [1, 3, 5] # L2 regularization
    }

    n_iter_search = 4
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        cv=3,
        scoring=SCORING,
        n_jobs=1, # CatBoost uses threads internally
        random_state=RANDOM_STATE,
        refit=True
    )

    print("Running CatBoost tuning...")
    # CatBoost might require specific dtypes, ensure train data is compatible
    # Typically handles float/int. Our preprocessing should be okay.
    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    print(f"Best CatBoost parameters found: {best_params}")
    best_cv_score = random_search.best_score_
    if SCORING.startswith('neg_'):
        best_cv_score = -best_cv_score
    print(f"Best CatBoost cross-validation score ({SCORING.replace('neg_','')+' (positive)'}): {best_cv_score:.4f}")

    best_model = random_search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    test_score = np.sqrt(mean_squared_error(y_test, y_pred_test))

    best_params_cleaned = {k.replace('cat__', ''): v for k, v in best_params.items()}

    return {
        'model': best_model,
        'best_params': best_params_cleaned,
        'cv_score': best_cv_score,
        'test_score': test_score,
        'backend': 'CatBoost'
    } 