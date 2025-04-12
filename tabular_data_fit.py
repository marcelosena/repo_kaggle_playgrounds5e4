import pandas as pd
import numpy as np
# Removed train_test_split, added KFold
from sklearn.model_selection import KFold
import models # Assumes models.py has updated function signatures
from tqdm import tqdm
import io
from sklearn.metrics import mean_squared_error
import collections # For defaultdict
import os # For creating directory
import pathlib # For path manipulation and directory creation
import argparse # For command-line arguments

# --- Parse Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Tabular data fitting with multiple models')
    
    # Root directory
    parser.add_argument('--root_dir', type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory of the project')
    
    # Dataset paths
    parser.add_argument('--train_path', type=str, default=None,
                        help='Path to training data CSV')
    parser.add_argument('--test_path', type=str, default=None, 
                        help='Path to competition test data CSV')
    
    # K-Fold parameter
    parser.add_argument('--k_folds', type=int, default=3,
                        help='Number of folds for cross-validation')
    
    # Model flags
    parser.add_argument('--run_gbt', action='store_true', default=True,
                        help='Run Gradient Boosting model')
    parser.add_argument('--no_gbt', action='store_false', dest='run_gbt',
                        help='Do not run Gradient Boosting model')
    
    parser.add_argument('--run_ridge', action='store_true', default=True,
                        help='Run Ridge Regression model')
    parser.add_argument('--no_ridge', action='store_false', dest='run_ridge',
                        help='Do not run Ridge Regression model')
    
    parser.add_argument('--run_xgb', action='store_true', default=True,
                        help='Run XGBoost model')
    parser.add_argument('--no_xgb', action='store_false', dest='run_xgb',
                        help='Do not run XGBoost model')
    
    parser.add_argument('--run_lgbm', action='store_true', default=True,
                        help='Run LightGBM model')
    parser.add_argument('--no_lgbm', action='store_false', dest='run_lgbm',
                        help='Do not run LightGBM model')
    
    parser.add_argument('--run_catboost', action='store_true', default=True,
                        help='Run CatBoost model')
    parser.add_argument('--no_catboost', action='store_false', dest='run_catboost',
                        help='Do not run CatBoost model')
    
    parser.add_argument('--run_fnn', action='store_true', default=True,
                        help='Run Feed-forward Neural Network model')
    parser.add_argument('--no_fnn', action='store_false', dest='run_fnn',
                        help='Do not run Feed-forward Neural Network model')
    
    parser.add_argument('--run_knn', action='store_true', default=True,
                        help='Run K-Nearest Neighbors model')
    parser.add_argument('--no_knn', action='store_false', dest='run_knn',
                        help='Do not run K-Nearest Neighbors model')
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.train_path is None:
        args.train_path = os.path.join(args.root_dir, "train.csv")
    if args.test_path is None:
        args.test_path = os.path.join(args.root_dir, "test.csv")
    
    return args

# --- Master Parameters ---
RANDOM_STATE = 42
TARGET_COLUMN = "Listening_Time_minutes"
ID_COLUMN = "id" # Column name for IDs in test.csv and submission

# --- Helper Functions ---
def ensure_dir_exists(file_path):
    """Ensures the directory for a file exists, creates it if not."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    return directory

# --- Data Loading ---
def load_data(path):
    """Loads tabular data from the specified CSV path."""
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        print(f"  Shape: {df.shape}")
        # Optional: Ensure target column is present early
        if TARGET_COLUMN not in df.columns and 'train' in path.lower():
             print(f"Error: Target column '{TARGET_COLUMN}' not found in loaded training data.")
             return pd.DataFrame()
        # Ensure ID column is present in test data
        if ID_COLUMN not in df.columns and 'test' in path.lower():
            print(f"Error: ID column '{ID_COLUMN}' not found in loaded competition test data.")
            # Allow continuing if ID is missing, but submission will fail later
            # return pd.DataFrame()
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return pd.DataFrame()

# --- Preprocessing within Fold ---
def preprocess_fold_data(df_train, df_test, target_col):
    """Preprocesses train/test data for a fold.
    Handles numeric conversion, categorical encoding (factorize).
    Returns processed df_train, df_test, numeric_cols list.
    """
    print("Preprocessing fold data...")
    train_proc = df_train.copy()
    test_proc = df_test.copy()
    encoding_maps = {}
    numeric_cols = []
    processed_feature_cols = [] # Track columns successfully processed

    # --- Process target column (if it exists in train_proc) ---
    if target_col in train_proc.columns:
        print(f"Processing target column '{target_col}'.")
        try:
            # Convert target, ensure it doesn't become NaN in train
            train_proc[target_col] = pd.to_numeric(train_proc[target_col])
            if train_proc[target_col].isnull().any():
                raise ValueError("Target column contains NaNs after numeric conversion in training data.")
            # Also process in test set if present
            if target_col in test_proc.columns:
                test_proc[target_col] = pd.to_numeric(test_proc[target_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Target column '{target_col}' processing error: {e}") from e
    else:
        print(f"Warning: Target column '{target_col}' not in training data for preprocessing.")

    # --- Process feature columns ---
    # Use columns present in train_proc, excluding the target
    feature_cols = [col for col in train_proc.columns if col != target_col]
    for col in feature_cols:
        print(f"Processing feature: '{col}'")
        try:
            # Try strict numeric conversion first
            train_proc[col] = pd.to_numeric(train_proc[col], errors='raise')
            # If successful, apply to test (coerce errors) and convert both to float
            if col in test_proc.columns:
                test_proc[col] = pd.to_numeric(test_proc[col], errors='coerce')
                test_proc[col] = test_proc[col].astype(float)
            train_proc[col] = train_proc[col].astype(float)
            numeric_cols.append(col)
            processed_feature_cols.append(col)
            print(f"  Processed '{col}' as numeric (float).")
        except (ValueError, TypeError):
            # If strict numeric fails, treat as categorical
            try:
                print(f"  Treating '{col}' as categorical.")
                train_proc[col] = train_proc[col].astype(str)
                if col in test_proc.columns:
                    test_proc[col] = test_proc[col].astype(str)

                # Use factorize on train data to get codes and mapping
                codes, unique_categories = pd.factorize(train_proc[col])
                encoding_map = {cat: code for code, cat in enumerate(unique_categories)}

                train_proc[col] = codes # Assign codes to train
                # Map test set using the learned mapping, fill unseen with -1
                if col in test_proc.columns:
                    test_proc[col] = test_proc[col].map(encoding_map).fillna(-1).astype(int)
                processed_feature_cols.append(col)
                print(f"  Successfully factorized '{col}'.")
            except Exception as e_cat:
                print(f"  Warning: Failed processing column '{col}' as categorical: {e_cat}. Skipping.")
                # Do not add to processed_feature_cols

    # --- Final Column Alignment ---
    # Keep only successfully processed feature columns + target (if target exists)
    final_cols_train = ([target_col] if target_col in train_proc.columns else []) + processed_feature_cols
    train_proc = train_proc[final_cols_train]
    
    # Align test set columns to match final train columns
    final_cols_test = [col for col in final_cols_train if col in test_proc.columns]
    test_proc = test_proc[final_cols_test]
    missing_in_test = [col for col in final_cols_train if col not in test_proc.columns]
    if missing_in_test:
        print(f"Warning: Columns missing in test set after processing: {missing_in_test}. Filling with 0.")
        for col_add in missing_in_test:
             test_proc[col_add] = 0 # Add missing columns filled with 0
    # Ensure final order matches train
    test_proc = test_proc[final_cols_train] 

    print("Preprocessing finished for fold.")
    return train_proc, test_proc, numeric_cols


# --- Main K-Fold Workflow ---
def main(args):
    """Main function: K-Fold CV for model evaluation & submission generation."""
    # Set up paths based on args
    ROOT_DIR = args.root_dir
    DATA_PATH = args.train_path
    COMPETITION_TEST_PATH = args.test_path
    K_FOLDS = args.k_folds
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")

    # Model flags from args
    RUN_GBT = args.run_gbt
    RUN_RIDGE = args.run_ridge
    RUN_XGB = args.run_xgb
    RUN_LGBM = args.run_lgbm
    RUN_CATBOOST = args.run_catboost
    RUN_FNN = args.run_fnn
    RUN_KNN = args.run_knn
    
    df_original = load_data(DATA_PATH)
    if df_original.empty: return

    # --- Optional: Print Stats of Original Data ---
    print("\n--- Basic Data Characteristics (Original Data) ---")
    print(f"Number of observations: {len(df_original)}")
    print(f"Number of features: {len(df_original.columns)}")
    print("\nFeature names:")
    print(df_original.columns.tolist())
    print("\nData types:")
    print(df_original.dtypes)
    print("\nMissing values per column:")
    print(df_original.isnull().sum())
    print("\nBasic statistics (numeric columns):")
    numeric_df_orig = df_original.select_dtypes(include=np.number)
    if not numeric_df_orig.empty: print(numeric_df_orig.describe())
    else: print("No numeric columns found for descriptive statistics.")
    print("------------------------------------\n")

    # --- Build Model Dictionary Based on Flags ---
    print("\n--- Configuring Models to Run ---")
    potential_models = {
        "GradientBoosting": (models.train_evaluate_gbt, RUN_GBT),
        "Ridge": (models.train_evaluate_ridge, RUN_RIDGE),
        "XGBoost": (models.train_evaluate_xgb, RUN_XGB),
        "LightGBM": (models.train_evaluate_lgbm, RUN_LGBM),
        "CatBoost": (models.train_evaluate_catboost, RUN_CATBOOST),
        "FNN": (models.train_evaluate_fnn, RUN_FNN),
        "KNN": (models.train_evaluate_knn, RUN_KNN)
    }
    
    models_to_run = {}
    for name, (func, should_run) in potential_models.items():
        if should_run:
            print(f"Including model: {name}")
            models_to_run[name] = func
        else:
            print(f"Excluding model: {name}")
    
    if not models_to_run:
        print("\nError: No models selected to run. Please enable at least one model flag.")
        return

    # --- K-Fold Setup ---
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_fold_results = collections.defaultdict(list) # Stores test scores
    # Store details from first fold for submission generation & summary
    fold_1_models = {}
    fold_1_params = {}
    fold_1_backends = {}

    print(f"--- Starting {K_FOLDS}-Fold Cross-Validation ---")

    # --- K-Fold Loop ---
    for fold, (train_index, test_index) in enumerate(kf.split(df_original)):
        print(f"\n===== Fold {fold + 1}/{K_FOLDS} =====")
        df_train_fold = df_original.iloc[train_index]
        df_test_fold = df_original.iloc[test_index]

        # --- Preprocess, Impute, Prepare Data for Fold ---
        try:
            df_train_proc, df_test_proc, num_cols = preprocess_fold_data(
                df_train_fold, df_test_fold, TARGET_COLUMN
            )

            # Drop NaNs from Training Data (important!)
            train_rows_before_drop = len(df_train_proc)
            df_train_proc.dropna(inplace=True)
            train_rows_dropped = train_rows_before_drop - len(df_train_proc)
            if train_rows_dropped > 0:
                 print(f"Dropped {train_rows_dropped} rows with NaNs from training data.")
            if df_train_proc.empty:
                raise ValueError("Training data empty after NaN drop.")

            # Impute NaNs in Test Data (essential!)
            if df_test_proc.isnull().sum().sum() > 0:
                print("Imputing NaNs in test data...")
                # Impute numeric with median from processed train fold
                for col in num_cols:
                    if col in df_test_proc.columns and df_test_proc[col].isnull().any():
                         median_val = df_train_proc[col].median() # Median from processed TRAIN data
                         if pd.isna(median_val): median_val = 0 # Handle case where train median is NaN
                         df_test_proc[col].fillna(median_val, inplace=True)

                # Impute categorical (factorized codes) with mode from processed train fold
                cat_cols = [c for c in df_train_proc.columns if c not in num_cols and c != TARGET_COLUMN]
                for col in cat_cols:
                    if col in df_test_proc.columns and df_test_proc[col].isnull().any():
                         mode_val = df_train_proc[col].mode()
                         mode_val = mode_val[0] if not mode_val.empty else -1 # Use -1 if mode is empty
                         df_test_proc[col].fillna(mode_val, inplace=True)

                # Final check: Drop rows if NaNs persist (shouldn't happen with above imputation)
                if df_test_proc.isnull().sum().sum() > 0:
                    print("Warning: NaNs still exist after imputation. Dropping remaining NaN rows from test.")
                    df_test_proc.dropna(inplace=True)

            if df_test_proc.empty:
                raise ValueError("Test data empty after NaN handling.")

            # Final data preparation for models
            X_train = df_train_proc.drop(columns=[TARGET_COLUMN])
            y_train = df_train_proc[TARGET_COLUMN]
            X_test = df_test_proc.drop(columns=[TARGET_COLUMN])
            y_test = df_test_proc[TARGET_COLUMN]

            # Final column check (should pass if preprocessing worked)
            if list(X_train.columns) != list(X_test.columns):
                 raise ValueError(f"Train/Test columns mismatch before training: {list(X_train.columns)} vs {list(X_test.columns)}")

        except Exception as e:
            print(f"ERROR preparing data for fold {fold + 1}: {e}. Skipping fold.")
            for model_name in models_to_run: all_fold_results[model_name].append(np.nan)
            all_fold_results['Ensemble'].append(np.nan)
            continue # Skip to next fold

        # --- Train/Evaluate Models for Fold --- 
        print(f"--- Training Models for Fold {fold + 1} ---")
        fold_individual_results = {} # Store results just for this fold for ensembling
        fold_tqdm = tqdm(models_to_run.items(), desc=f"Fold {fold+1} Models", leave=False)
        for model_name, train_func in fold_tqdm:
            fold_tqdm.set_description(f"Fold {fold+1} - {model_name}")
            try:
                model_results = train_func(X_train, y_train, X_test, y_test)
                test_score = model_results.get('test_score', np.nan)
                all_fold_results[model_name].append(test_score)
                fold_individual_results[model_name] = model_results # Store full result
                print(f"  {model_name} Fold {fold+1} Score (RMSE): {test_score:.4f}")
                if fold == 0: # Store details from first fold only
                    fold_1_models[model_name] = model_results.get('model')
                    fold_1_params[model_name] = model_results.get('best_params', 'N/A')
                    fold_1_backends[model_name] = model_results.get('backend', 'N/A')
            except Exception as e:
                print(f"ERROR training/evaluating {model_name} in fold {fold + 1}: {e}")
                all_fold_results[model_name].append(np.nan)
                fold_individual_results[model_name] = None # Indicate failure for ensemble
                if fold == 0:
                     fold_1_models[model_name] = 'Error'
                     fold_1_params[model_name] = 'Error'
                     fold_1_backends[model_name] = 'Error'
        
        # --- Ensemble Calculation for Fold --- 
        print(f"--- Generating Ensemble for Fold {fold + 1} ---")
        fold_preds = []
        fold_valid_models = []
        for model_name, result in fold_individual_results.items():
            if result is not None and pd.notna(result.get('test_score')): 
                fold_valid_models.append(model_name)
                backend = result.get('backend', 'sklearn')
                model_obj = result.get('model')
                try:
                    if model_name == 'FNN':
                        preds = model_obj.predict(X_test.astype(np.float32))
                    elif model_name == 'KNN' and backend == 'cuml':
                        cuml_model = model_obj['model']
                        scaler = model_obj['scaler']
                        X_test_gpu = models.cudf.DataFrame.from_pandas(X_test.astype(np.float32))
                        X_test_scaled = scaler.transform(X_test_gpu)
                        preds_gpu = cuml_model.predict(X_test_scaled)
                        preds = preds_gpu.to_numpy()
                    else: # Sklearn pipelines
                        preds = model_obj.predict(X_test)
                    fold_preds.append(preds.flatten())
                except Exception as e_pred:
                     print(f"Error generating prediction for {model_name} in fold {fold+1} ensemble: {e_pred}")
                     # Remove model from list if prediction failed
                     if model_name in fold_valid_models: fold_valid_models.remove(model_name)
            else:
                 print(f"Skipping failed model {model_name} for fold {fold+1} ensemble.")

        if fold_preds:
            stacked_preds = np.vstack(fold_preds)
            avg_preds = np.mean(stacked_preds, axis=0)
            y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
            ensemble_score = np.sqrt(mean_squared_error(y_test_np, avg_preds))
            print(f"  Ensemble Fold {fold+1} Score (RMSE) from models {fold_valid_models}: {ensemble_score:.4f}")
            all_fold_results['Ensemble'].append(ensemble_score)
        else:
             print(f"No successful models/predictions for ensemble in fold {fold+1}.")
             all_fold_results['Ensemble'].append(np.nan)

    # --- Aggregate and Summarize Results --- 
    print("\n--- Overall Performance Summary (Across Folds) ---")
    summary_data = []
    # Ensure Ensemble is included only if models were run
    model_names_for_summary = list(models_to_run.keys()) 
    if 'Ensemble' in all_fold_results:
         model_names_for_summary += ['Ensemble']
    
    for model_name in model_names_for_summary:
        scores = all_fold_results[model_name]
        valid_scores = [s for s in scores if pd.notna(s)]
        mean_score = np.mean(valid_scores) if valid_scores else np.nan
        std_score = np.std(valid_scores) if valid_scores else np.nan
        summary_str = f"{mean_score:.4f} +/- {std_score:.4f}" if valid_scores else "Failed"

        summary_data.append({
            'Model': model_name,
            'Backend': fold_1_backends.get(model_name, 'Ensemble' if model_name == 'Ensemble' else 'N/A'),
            'Params (Fold 1)': str(fold_1_params.get(model_name, 'N/A' if model_name != 'Ensemble' else 'Equal Weight Average')),
            f'Mean Score ({K_FOLDS}-Fold RMSE)': mean_score,
            f'Std Dev ({K_FOLDS}-Fold RMSE)': std_score,
            'Score Summary': summary_str
        })

    summary_df = pd.DataFrame(summary_data).set_index('Model')
    summary_df.sort_values(f'Mean Score ({K_FOLDS}-Fold RMSE)', inplace=True, na_position='last')
    display_cols = ['Backend', 'Score Summary', 'Params (Fold 1)']
    print(summary_df[display_cols])

    # --- Save Summary Results to CSV ---
    print("\n--- Saving Performance Summary ---")
    results_path = os.path.join(RESULTS_DIR, "model_performance_summary.csv")
    ensure_dir_exists(results_path)
    
    try:
        # Save with index=True because the index contains the Model names
        summary_df.to_csv(results_path, index=True)
        print(f"Performance summary saved to: {results_path}")
    except Exception as e:
        print(f"Error saving performance summary to {results_path}: {e}")

    # --- Submission File Generation ---
    print("\n--- Generating Submission File ---")
    # Find best *individual* model among those that actually ran
    valid_individual_summary = summary_df[(summary_df.index != 'Ensemble') & (summary_df.index.isin(models_to_run.keys()))].dropna(subset=[f'Mean Score ({K_FOLDS}-Fold RMSE)'])
    if valid_individual_summary.empty:
        print("No individual models succeeded across folds. Cannot determine best model for submission.")
        return
    best_model_name = valid_individual_summary[f'Mean Score ({K_FOLDS}-Fold RMSE)'].idxmin()
    print(f"Best individual model (based on mean CV score): {best_model_name}")

    best_model_obj_fold1 = fold_1_models.get(best_model_name)
    best_model_backend = fold_1_backends.get(best_model_name)

    if best_model_obj_fold1 is None or best_model_obj_fold1 == 'Error':
        print(f"Error: Model object for the best model ('{best_model_name}') from Fold 1 is not available. Cannot generate submission.")
        return

    # Load competition test data
    df_comp_test_orig = load_data(COMPETITION_TEST_PATH)
    if df_comp_test_orig.empty:
        print("Error: Failed to load competition test data. Cannot generate submission.")
        return
    # Ensure ID column exists before trying to access it
    if ID_COLUMN not in df_comp_test_orig.columns:
        print(f"Error: ID column '{ID_COLUMN}' not found in competition test data '{COMPETITION_TEST_PATH}'. Cannot generate submission.")
        return
    comp_test_ids = df_comp_test_orig[ID_COLUMN]

    # Preprocess competition test data using the *full original training data* for fitting transforms
    print("\nPreprocessing competition test data based on full training data...")
    try:
        # Note: preprocess_fold_data needs target_col for alignment, even if not present in test
        # We pass df_original (train) and df_comp_test_orig (test)
        df_train_full_proc, df_comp_test_proc, num_cols_full = preprocess_fold_data(
            df_original, df_comp_test_orig, TARGET_COLUMN
        )
        
        # Impute NaNs in the processed competition test data
        if df_comp_test_proc.isnull().sum().sum() > 0:
            print("Imputing NaNs in processed competition test data...")
            # Need processed full train data for imputation stats
            df_train_full_proc.dropna(inplace=True)
            if df_train_full_proc.empty:
                 raise ValueError("Full training data is empty after NaN drop, cannot impute test data.")
                 
            for col in num_cols_full:
                if col in df_comp_test_proc.columns and df_comp_test_proc[col].isnull().any():
                    median_val = df_train_full_proc[col].median()
                    if pd.isna(median_val): median_val = 0
                    df_comp_test_proc[col].fillna(median_val, inplace=True)
            cat_cols_full = [c for c in df_train_full_proc.columns if c not in num_cols_full and c != TARGET_COLUMN]
            for col in cat_cols_full:
                 if col in df_comp_test_proc.columns and df_comp_test_proc[col].isnull().any():
                     mode_val = df_train_full_proc[col].mode()
                     mode_val = mode_val[0] if not mode_val.empty else -1
                     df_comp_test_proc[col].fillna(mode_val, inplace=True)
            
            if df_comp_test_proc.isnull().sum().sum() > 0:
                 print("Warning: NaNs remain in competition test data after imputation. Submission might be incomplete/incorrect.")
                 # Decide whether to drop or fill remaining NaNs with a default (e.g., 0)
                 # df_comp_test_proc.fillna(0, inplace=True) 

        # Prepare final competition test features (ensure correct columns, drop target if present)
        final_features = [col for col in df_train_full_proc.columns if col != TARGET_COLUMN]
        if TARGET_COLUMN in df_comp_test_proc.columns:
             X_comp_test_final = df_comp_test_proc.drop(columns=[TARGET_COLUMN])
        else:
             X_comp_test_final = df_comp_test_proc
        
        # Ensure test features match final feature list
        X_comp_test_final = X_comp_test_final[final_features]
        
        if X_comp_test_final.empty:
             raise ValueError("Competition test data is empty after processing.")

    except Exception as e:
        print(f"Error preprocessing or imputing competition test data: {e}. Cannot generate submission.")
        return

    # Generate predictions using the best model from Fold 1
    print(f"Generating predictions using {best_model_name} (from Fold 1)...")
    try:
        if best_model_name == 'FNN':
            predictions = best_model_obj_fold1.predict(X_comp_test_final.astype(np.float32))
        elif best_model_name == 'KNN' and best_model_backend == 'cuml':
            cuml_model = best_model_obj_fold1['model']
            scaler = best_model_obj_fold1['scaler'] # Scaler fitted on Fold 1 train data
            # This scaler was fit on Fold 1 data, not full data. Re-scaling might be needed based on full data.
            # For consistency, ideally refit scaler on full train data or use Fold 1 scaler carefully.
            # Using Fold 1 scaler here for simplicity, assuming distribution is similar.
            print("Warning: Using KNN/cuML scaler from Fold 1 for submission predictions.")
            X_comp_test_gpu = models.cudf.DataFrame.from_pandas(X_comp_test_final.astype(np.float32))
            X_comp_test_scaled = scaler.transform(X_comp_test_gpu)
            preds_gpu = cuml_model.predict(X_comp_test_scaled)
            predictions = preds_gpu.to_numpy()
        else: # Sklearn pipelines (already include scaler fitted on fold 1)
            # These pipelines were fitted on Fold 1 data. Ideally, refit best model on FULL training data.
            # Using Fold 1 model here for simplicity.
            print("Warning: Using model pipeline fitted on Fold 1 data for submission predictions.")
            predictions = best_model_obj_fold1.predict(X_comp_test_final)
        
        # Ensure predictions are non-negative (if applicable for target)
        predictions[predictions < 0] = 0
        
    except Exception as e:
        print(f"Error generating predictions for submission: {e}. Cannot generate submission.")
        return

    # Create submission directory if it doesn't exist
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    ensure_dir_exists(submission_path)

    # Create submission DataFrame
    # Ensure IDs and predictions align, especially if rows were dropped from test during imputation
    if len(comp_test_ids) != len(predictions):
         print(f"Warning: Length mismatch between IDs ({len(comp_test_ids)}) and predictions ({len(predictions)}). Check imputation/row dropping.")
         # Attempt to align based on index if possible (risky)
         # This assumes comp_test_ids index matches X_comp_test_final index
         try:
             aligned_ids = comp_test_ids.loc[X_comp_test_final.index]
             if len(aligned_ids) == len(predictions):
                  comp_test_ids = aligned_ids
                  print("Aligned IDs to predictions based on index.")
             else:
                  print("Error: Cannot align IDs and predictions. Submission file not generated.")
                  return
         except Exception as align_e:
             print(f"Error aligning IDs: {align_e}. Submission file not generated.")
             return
             
    submission_df = pd.DataFrame({
        ID_COLUMN: comp_test_ids,
        TARGET_COLUMN: predictions.flatten() # Ensure predictions are 1D
    })

    # Save submission file
    try:
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to: {submission_path}")
    except Exception as e:
        print(f"Error saving submission file: {e}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 