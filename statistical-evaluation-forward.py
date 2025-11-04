#The code for the research presented in the paper titled "A_deep_learning_method_for_extinction_spectrum_prediction_and_graphene-coated_silver_nanoparticles_inverse_design"

#This code corresponds to the article's Statistical evaluation of forward Deep Neural Network (DNN) section.
#Please cite the paper in any publication using this code.

# ===========================
# Comprehensive Training Data Analysis - Separate Plots with Preprocessors Recreation
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime
import json
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# ===========================
# Font Configuration - Times New Roman
# ===========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 10
plt.rcParams['legend.fontsize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 28

# ===========================
# Paths Configuration
# ===========================
BASE_DIR = Path("./nn_regression_ga_outputs")
PLOTS_DIR = BASE_DIR / "plots"
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "best_model"
PLOTS_DIR.mkdir(exist_ok=True)

# ===========================
# Data Loading & Preprocessing Functions
# ===========================
def load_dataframe(csv_path: Path):
    """Load dataframe and separate features and target"""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :3].copy()
    y = df.iloc[:, -1].astype(float).copy()
    return X, y, df

def correct_targets(y: pd.Series):
    """Apply the same correction as in training"""
    y_corr = y.copy()
    y_corr[y_corr < 0] = -y_corr[y_corr < 0]
    y_corr = np.minimum(y_corr, 1.0)
    return y_corr

def preprocess_inputs(X_train, X_val, cfg):
    """SAME AS ORIGINAL - Create preprocessors from scratch"""
    if cfg["c_encoding"] == "numeric":
        scaler = StandardScaler().fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_val), {"scaler": scaler, "encoder": None}
    else:
        enc = OneHotEncoder(sparse=False, categories="auto")
        C_tr = enc.fit_transform(X_train[:, 2].reshape(-1,1))
        C_va = enc.transform(X_val[:, 2].reshape(-1,1))
        AB_tr = X_train[:, :2]
        AB_va = X_val[:, :2]
        scaler = StandardScaler().fit(AB_tr)
        AB_tr_s = scaler.transform(AB_tr)
        AB_va_s = scaler.transform(AB_va)
        return np.hstack([AB_tr_s, C_tr]), np.hstack([AB_va_s, C_va]), {"scaler": scaler, "encoder": enc}

def preprocess_test_data(X_test, cfg, preprocessors):
    """Preprocess test data using preprocessors"""
    if cfg["c_encoding"] == "numeric":
        scaler = preprocessors["scaler"]
        X_test_p = scaler.transform(X_test.values)
    else:
        scaler, enc = preprocessors["scaler"], preprocessors["encoder"]
        C_te = enc.transform(X_test.values[:, 2].reshape(-1,1))
        AB_te = scaler.transform(X_test.values[:, :2])
        X_test_p = np.hstack([AB_te, C_te])
    return X_test_p

# ===========================
# Load Required Resources with Preprocessors Recreation
# ===========================
def load_required_resources():
    """Load model, config and recreate preprocessors"""
    print("🔄 Loading resources...")
    
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_DIR / "trained_model.h5", compile=False)
        print("✅ Model loaded successfully")
        
        # Load config
        with open(ARTIFACTS / "best_config.json", "r") as f:
            best_cfg = json.load(f)
        print("✅ Config loaded successfully")
        
        # Load training data to recreate preprocessors
        TRAIN_CSV = Path("C:\\Users\\hoseini\\Desktop\\merge-csv.com__68d6b5a27ee3c.csv")
        X_all, y_raw, _ = load_dataframe(TRAIN_CSV)
        y_all = correct_targets(y_raw)
        
        # Recreate the same train/validation split as original training
        X_train, X_val, y_train, y_val = train_test_split(
            X_all.values, y_all.values, test_size=0.10, random_state=42, shuffle=True
        )
        
        # Recreate preprocessors using the same function as training
        _, _, preprocessors = preprocess_inputs(X_train, X_val, best_cfg)
        print("✅ Preprocessors recreated successfully")
        
        return model, best_cfg, preprocessors
        
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return None, None, None


model, best_cfg, preprocessors = load_required_resources()

if model is None or best_cfg is None or preprocessors is None:
    print("❌ Failed to load resources.")
else:
    # ===========================
    # Analysis Functions
    # ===========================
    def comprehensive_training_analysis(model, X_train, y_train, X_val, y_val, best_cfg, preprocessors):
        """
        Perform comprehensive analysis on training data
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE TRAINING DATA ANALYSIS")
        print("="*70)
        
        # Preprocess training and validation data
        X_train_p = preprocess_test_data(X_train, best_cfg, preprocessors)
        X_val_p = preprocess_test_data(X_val, best_cfg, preprocessors)
        
        # Generate predictions
        y_train_pred = model.predict(X_train_p, verbose=0).ravel()
        y_val_pred = model.predict(X_val_p, verbose=0).ravel()
        
        # Calculate metrics for both sets
        train_metrics = calculate_detailed_metrics(y_train, y_train_pred, "Training")
        val_metrics = calculate_detailed_metrics(y_val, y_val_pred, "Validation")
        
        # Compare training vs validation performance
        print("\n📊 TRAINING vs VALIDATION COMPARISON")
        print("=" * 50)
        print(f"{'Metric':<20} {'Training':<12} {'Validation':<12} {'Gap':<10}")
        print("-" * 50)
        
        metrics_to_compare = ['mse', 'rmse', 'mae', 'r2', 'mape']
        for metric in metrics_to_compare:
            train_val = train_metrics[metric]
            val_val = val_metrics[metric]
            gap = abs(train_val - val_val)
            
            if metric in ['r2']:
                print(f"{metric.upper():<20} {train_val:.4f}      {val_val:.4f}      {gap:.4f}")
            else:
                print(f"{metric.upper():<20} {train_val:.6f}  {val_val:.6f}  {gap:.6f}")
        
        # Detect overfitting
        overfitting_threshold = 0.05  # 5% gap in R²
        r2_gap = abs(train_metrics['r2'] - val_metrics['r2'])
        
        if r2_gap > overfitting_threshold:
            print(f"⚠️  Potential overfitting detected: R² gap = {r2_gap:.4f} (> {overfitting_threshold})")
        else:
            print(f"✅ Good generalization: R² gap = {r2_gap:.4f} (≤ {overfitting_threshold})")
        
        return train_metrics, val_metrics, y_train_pred, y_val_pred

    def calculate_detailed_metrics(y_true, y_pred, dataset_name):
        """Calculate comprehensive metrics for any dataset"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage errors (handle near-zero values)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
            mape = np.nan_to_num(mape, nan=100.0, posinf=100.0)
        
        residuals = y_true - y_pred
        
        print(f"\n📈 {dataset_name} Set Metrics:")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return {
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "mean_residual": np.mean(residuals), "std_residual": np.std(residuals)
        }

    def create_individual_training_plots(y_train_true, y_train_pred, y_val_true, y_val_pred, train_metrics, val_metrics):
        """Create SEPARATE plots for training analysis"""
        
        print(f"\n📊 Creating Individual Training Analysis Plots...")
        
        # 1. Training Set - True vs Predicted (Individual)
        plt.figure(figsize=(12, 10))
        plt.scatter(y_train_true, y_train_pred, alpha=0.7, s=80, color='blue', edgecolors='black', linewidth=0.5, label='Training Data')
        
        min_val = min(y_train_true.min(), y_train_pred.min())
        max_val = max(y_train_true.max(), y_train_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=3, label='Perfect Prediction')
        
        plt.xlabel('True Values', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Predicted Values', fontname='Times New Roman', fontsize=28)
        plt.title('Training Set: True vs Predicted Values', fontname='Times New Roman', fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'R² = {train_metrics["r2"]:.4f}\nRMSE = {train_metrics["rmse"]:.4f}\nMAE = {train_metrics["mae"]:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=20, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_true_vs_predicted.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 2. Validation Set - True vs Predicted (Individual)
        plt.figure(figsize=(12, 10))
        plt.scatter(y_val_true, y_val_pred, alpha=0.7, s=80, color='green', edgecolors='black', linewidth=0.5, label='Validation Data')
        
        min_val = min(y_val_true.min(), y_val_pred.min())
        max_val = max(y_val_true.max(), y_val_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', linewidth=3, label='Perfect Prediction')
        
        plt.xlabel('True Values', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Predicted Values', fontname='Times New Roman', fontsize=28)
        plt.title('Validation Set: True vs Predicted Values', fontname='Times New Roman', fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'R² = {val_metrics["r2"]:.4f}\nRMSE = {val_metrics["rmse"]:.4f}\nMAE = {val_metrics["mae"]:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=20, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "validation_true_vs_predicted.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 3. Residuals Comparison Plot (Individual)
        plt.figure(figsize=(12, 8))
        train_residuals = y_train_true - y_train_pred
        val_residuals = y_val_true - y_val_pred
        
        plt.hist(train_residuals, bins=30, alpha=0.7, color='blue', density=True, label='Training', edgecolor='black')
        plt.hist(val_residuals, bins=30, alpha=0.7, color='green', density=True, label='Validation', edgecolor='black')
        plt.xlabel('Residuals', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Density', fontname='Times New Roman', fontsize=28)
        plt.title('Residuals Distribution: Training vs Validation', fontname='Times New Roman', fontsize=30)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "residuals_distribution_comparison.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 4. MAE by Value Range with Scientific Notation (Individual)
        plt.figure(figsize=(14, 8))
        
        error_ranges = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        train_errors_by_range = []
        val_errors_by_range = []
        
        for i in range(len(error_ranges)-1):
            train_mask = (y_train_true >= error_ranges[i]) & (y_train_true < error_ranges[i+1])
            val_mask = (y_val_true >= error_ranges[i]) & (y_val_true < error_ranges[i+1])
            
            if train_mask.sum() > 0:
                train_mae = mean_absolute_error(y_train_true[train_mask], y_train_pred[train_mask])
            else:
                train_mae = 0
                
            if val_mask.sum() > 0:
                val_mae = mean_absolute_error(y_val_true[val_mask], y_val_pred[val_mask])
            else:
                val_mae = 0
                
            train_errors_by_range.append(train_mae)
            val_errors_by_range.append(val_mae)
        
        x_pos = np.arange(len(error_ranges)-1)
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, train_errors_by_range, width, label='Training', color='blue', alpha=0.7, edgecolor='black')
        bars2 = plt.bar(x_pos + width/2, val_errors_by_range, width, label='Validation', color='green', alpha=0.7, edgecolor='black')
        
        plt.xlabel('Value Range', fontname='Times New Roman', fontsize=28)
        plt.ylabel('MAE', fontname='Times New Roman', fontsize=28)
        plt.title('MAE by Value Range', fontname='Times New Roman', fontsize=30)
        plt.xticks(x_pos, [f'{error_ranges[i]:.2f}-{error_ranges[i+1]:.2f}' for i in range(len(error_ranges)-1)], 
                   rotation=45, fontname='Times New Roman', fontsize=24)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set scientific notation for y-axis
        plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # Add value labels on bars with scientific notation
        for bars, errors in zip([bars1, bars2], [train_errors_by_range, val_errors_by_range]):
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{error:.2e}', ha='center', va='bottom', fontsize=16, 
                            fontname='Times New Roman')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mae_by_value_range.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 5. Metrics Comparison Bar Plot (Individual)
        plt.figure(figsize=(14, 8))
        
        metrics_names = ['MSE', 'RMSE', 'MAE', 'R²']
        train_metrics_vals = [train_metrics['mse'], train_metrics['rmse'], train_metrics['mae'], train_metrics['r2']]
        val_metrics_vals = [val_metrics['mse'], val_metrics['rmse'], val_metrics['mae'], val_metrics['r2']]
        
        x_pos = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, train_metrics_vals, width, label='Training', 
                       color='blue', alpha=0.7, edgecolor='black')
        bars2 = plt.bar(x_pos + width/2, val_metrics_vals, width, label='Validation', 
                       color='green', alpha=0.7, edgecolor='black')
        
        plt.xlabel('Metrics', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Values', fontname='Times New Roman', fontsize=28)
        plt.title('Training vs Validation Metrics Comparison', fontname='Times New Roman', fontsize=30)
        plt.xticks(x_pos, metrics_names, fontname='Times New Roman', fontsize=24)
        plt.legend(fontsize=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=16, 
                        fontname='Times New Roman')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metrics_comparison.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 6. Residuals vs Predicted Values (Individual - Training)
        plt.figure(figsize=(12, 8))
        plt.scatter(y_train_pred, train_residuals, alpha=0.6, s=60, color='blue')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=3)
        plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Residuals', fontname='Times New Roman', fontsize=28)
        plt.title('Training Set: Residuals vs Predicted Values', fontname='Times New Roman', fontsize=30)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_residuals_vs_predicted.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        # 7. Residuals vs Predicted Values (Individual - Validation)
        plt.figure(figsize=(12, 8))
        plt.scatter(y_val_pred, val_residuals, alpha=0.6, s=60, color='green')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=3)
        plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=28)
        plt.ylabel('Residuals', fontname='Times New Roman', fontsize=28)
        plt.title('Validation Set: Residuals vs Predicted Values', fontname='Times New Roman', fontsize=30)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "validation_residuals_vs_predicted.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()

    def analyze_feature_importance_training(model, X_train, best_cfg, preprocessors, feature_names=['A', 'B', 'C']):
        """Analyze feature importance using training data"""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS (Training Data)")
        print("=" * 50)
        
        # Preprocess training data
        X_train_p = preprocess_test_data(X_train, best_cfg, preprocessors)
        
        # Get baseline performance
        y_train_pred = model.predict(X_train_p, verbose=0)
        baseline_mse = mean_squared_error(X_train.iloc[:, -1] if hasattr(X_train, 'iloc') else X_train[:, -1], 
                                        y_train_pred)
        
        importance_scores = {}
        
        for i, feature in enumerate(feature_names):
            # Permutation importance
            X_perturbed = X_train_p.copy()
            np.random.shuffle(X_perturbed[:, i])
            
            perturbed_pred = model.predict(X_perturbed, verbose=0)
            perturbed_mse = mean_squared_error(X_train.iloc[:, -1] if hasattr(X_train, 'iloc') else X_train[:, -1], 
                                             perturbed_pred)
            
            importance = (perturbed_mse - baseline_mse) / baseline_mse * 100
            importance_scores[feature] = importance
            
            print(f"   {feature}: {importance:+.2f}% MSE increase when shuffled")
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        features = list(importance_scores.keys())
        importance_vals = list(importance_scores.values())
        
        bars = ax.barh(features, importance_vals, color='skyblue', edgecolor='black')
        ax.set_xlabel('Importance (% MSE Increase)', fontname='Times New Roman', fontsize=16)
        ax.set_title('Feature Importance (Permutation Method)', fontname='Times New Roman', fontsize=18)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_vals):
            ax.text(bar.get_width() + max(importance_vals)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}%', ha='left', va='center', fontname='Times New Roman', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance_training.tiff", dpi=600, format="tiff", bbox_inches='tight')
        plt.show()
        
        return importance_scores

    # ===========================
    # Main Execution
    # ===========================
    def load_training_data():
        """Load training data for analysis"""
        TRAIN_CSV = Path("C:\\Users\\hoseini\\Desktop\\merge-csv.com__68d6b5a27ee3c.csv")
        X_all, y_raw, df_train = load_dataframe(TRAIN_CSV)
        y_all = correct_targets(y_raw)
        
        # Recreate the same train/validation split as original training
        X_train, X_val, y_train, y_val = train_test_split(
            X_all.values, y_all.values, test_size=0.10, random_state=42, shuffle=True
        )
        
        return X_train, X_val, y_train, y_val, df_train

    # اجرای اصلی
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE TRAINING DATA ANALYSIS")
    print("="*70)

    # Load training data
    X_train, X_val, y_train, y_val, df_train = load_training_data()

    # Perform comprehensive training analysis
    train_metrics, val_metrics, y_train_pred, y_val_pred = comprehensive_training_analysis(
        model, pd.DataFrame(X_train), y_train, pd.DataFrame(X_val), y_val, best_cfg, preprocessors
    )

    # Create INDIVIDUAL training analysis plots
    create_individual_training_plots(y_train, y_train_pred, y_val, y_val_pred, train_metrics, val_metrics)

    # Analyze feature importance
    feature_importance = analyze_feature_importance_training(
        model, pd.DataFrame(X_train), best_cfg, preprocessors
    )

    # Save training analysis report
    training_report = {
        "timestamp": datetime.now().isoformat(),
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "generalization_gap": {
            "r2_gap": abs(train_metrics['r2'] - val_metrics['r2']),
            "mse_gap": abs(train_metrics['mse'] - val_metrics['mse']),
            "mae_gap": abs(train_metrics['mae'] - val_metrics['mae'])
        },
        "feature_importance": feature_importance,
        "overfitting_assessment": "Potential overfitting" if abs(train_metrics['r2'] - val_metrics['r2']) > 0.05 else "Good generalization"
    }

    with open(BASE_DIR / "training_analysis_report.json", "w") as f:
        json.dump(training_report, f, indent=2)

    print(f"\n✅ Training analysis completed!")
    print(f"📊 Report saved to: {BASE_DIR / 'training_analysis_report.json'}")
    print(f"📈 Individual plots saved to: {PLOTS_DIR}")