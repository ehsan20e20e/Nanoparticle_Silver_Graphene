#The code for the research presented in the paper titled "A_deep_learning_method_for_extinction_spectrum_prediction_and_graphene-coated_silver_nanoparticles_inverse_design"

#This code corresponds to the article's invers Deep Neural Network (DNN) section.
#Please cite the paper in any publication using this code.

# ===========================
# Setup
# ===========================
import os, json, random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


try:
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    print("✅ TensorFlow imported successfully")
    
   
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    
except ImportError:
    print("Installing TensorFlow...")
    !pip install -q tensorflow==2.18.0
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

# تنظیم seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ===========================
# Paths
# ===========================
BASE_DIR = Path("./nn_inverse_ga_outputs")
BASE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("C:\\Users\\hoseini\\Desktop\\merge-csv.com__68d6b5a27ee3c.csv")
TEST_CSV = Path("C:\\Users\\hoseini\\Desktop\\Test_Forward_model.csv") 

MODEL_DIR = BASE_DIR / "best_model"
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ===========================
# Data Loading & Preprocessing for INVERSE
# ===========================
def load_and_prepare_inverse_data(csv_path: Path, sequence_length=600):
    """upload and prepare dataset"""
    df = pd.read_csv(csv_path)
    
    
    print("data structure:")
    print(df.head())
    print(f"number of rows: {len(df)}")
    print(f"column: {df.columns.tolist()}")
    
   
    if len(df.columns) >= 4:
        df.columns = ['index', 'param1', 'param2', 'output']
    else:
       
        df.columns = ['param1', 'param2', 'output'] + [f'extra_{i}' for i in range(len(df.columns)-3)]
    
   
    def preprocess_outputs(outputs):
        outputs = np.array(outputs, dtype=float)
        outputs = np.abs(outputs)  
        outputs = np.clip(outputs, 0, 1)  
        return outputs
    
    df['output_processed'] = preprocess_outputs(df['output'].values)
    
    
    X_inverse, y_inverse = [], []
    
    
    grouped = df.groupby(['param1', 'param2'])
    
    for (param1, param2), group in grouped:
        
        if 'index' in group.columns:
            group = group.sort_values('index')
        else:
            group = group.sort_index()
        
        outputs = group['output_processed'].values
        
        
        if len(outputs) >= sequence_length:
          
            for i in range(0, len(outputs) - sequence_length + 1, sequence_length):
                X_inverse.append(outputs[i:i+sequence_length])
                y_inverse.append([param1, param2])
    
    X_inverse = np.array(X_inverse)
    y_inverse = np.array(y_inverse)
    
    print(f"invers data: {X_inverse.shape} -> {y_inverse.shape}")
    print(f"number of samples: {len(X_inverse)}")
    return X_inverse, y_inverse, df


X_all, y_all, df_original = load_and_prepare_inverse_data(TRAIN_CSV, sequence_length=600)

if len(X_all) == 0:
    print("❌ not enough data!")
    exit()


X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.10, random_state=42, shuffle=False  
)

print(f"training data: {X_train.shape}, {y_train.shape}")
print(f"validation data : {X_val.shape}, {y_val.shape}")

# ===========================
# Preprocessing for INVERSE
# ===========================
def preprocess_inverse_inputs(X_train, X_val):
    """ preprocessing invers input data"""
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler_X

def preprocess_inverse_outputs(y_train, y_val):
    """preprocessing invers output data"""
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    return y_train_scaled, y_val_scaled, scaler_y


X_train_p, X_val_p, scaler_X = preprocess_inverse_inputs(X_train, X_val)
y_train_p, y_val_p, scaler_y = preprocess_inverse_outputs(y_train, y_val)

# ===========================

# ===========================
def build_inverse_model(input_dim, cfg):
    """creat invers modle"""
    model = tf.keras.Sequential()
    n_layers = cfg["n_layers"]
    units_list = cfg["units_per_layer"]
    
    
    if len(units_list) != n_layers:
        units_list = [512, 256, 128] 
    
    
    model.add(layers.Dense(units_list[0], activation=cfg["activation"], 
                          kernel_regularizer=regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None,
                          input_shape=(input_dim,)))
    if cfg["dropout"] > 0:
        model.add(layers.Dropout(cfg["dropout"]))
    
    
    for i in range(1, n_layers):
        units = units_list[i]
        model.add(layers.Dense(units, activation=cfg["activation"],
                              kernel_regularizer=regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None))
        if cfg["dropout"] > 0:
            model.add(layers.Dropout(cfg["dropout"]))
    
    
    model.add(layers.Dense(2, activation="linear"))
    
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss="mse", metrics=["mse", "mae"])
    return model

# ===========================

# ===========================
ACTIVATIONS = ["relu", "tanh", "elu"]

def random_inverse_config():
    """random setting"""
    n_layers = random.randint(3, 6)  
    
     
    if n_layers == 3:
        units = [random.choice([512, 600, 700]), 256, 128]
    elif n_layers == 4:
        units = [random.choice([600, 700, 800]), 400, 200, 100]
    elif n_layers == 5:
        units = [random.choice([700, 800, 900]), 500, 300, 150, 75]
    else:  # 6 layers
        units = [random.choice([800, 900, 1024]), 600, 400, 200, 100, 50]
    
    return {
        "n_layers": n_layers,
        "units_per_layer": units,
        "activation": random.choice(ACTIVATIONS),
        "dropout": round(random.uniform(0.1, 0.4), 2),
        "l2": round(random.choice([1e-5, 1e-4, 5e-4]), 6),
        "lr": 10**random.uniform(-4, -3),
        "batch_size": random.choice([8, 16, 32]),  
        "epochs": 200,
        "patience": 20,
    }

def mutate_inverse(cfg, p=0.3):
    """jump for invers"""
    new_cfg = json.loads(json.dumps(cfg))
    
    if random.random() < p:
        new_cfg["activation"] = random.choice(ACTIVATIONS)
    if random.random() < p:
        new_cfg["dropout"] = round(np.clip(new_cfg["dropout"] + random.uniform(-0.1, 0.1), 0.05, 0.5), 2)
    if random.random() < p:
        new_cfg["l2"] = round(random.choice([1e-5, 1e-4, 5e-4]), 6)
    if random.random() < p:
        new_cfg["lr"] = 10**random.uniform(-4, -3)
    if random.random() < p:
        new_cfg["batch_size"] = random.choice([8, 16, 32])
    if random.random() < p:
        nl = random.randint(3, 6)
        
        if nl == 3:
            units = [random.choice([512, 600, 700]), 256, 128]
        elif nl == 4:
            units = [random.choice([600, 700, 800]), 400, 200, 100]
        elif nl == 5:
            units = [random.choice([700, 800, 900]), 500, 300, 150, 75]
        else:  # 6 layers
            units = [random.choice([800, 900, 1024]), 600, 400, 200, 100, 50]
        
        new_cfg["n_layers"] = nl
        new_cfg["units_per_layer"] = units
    
    return new_cfg

def crossover_inverse(a, b):
   
    child = {}
    for k in a.keys():
        child[k] = random.choice([a[k], b[k]])
    
   
    nl = child["n_layers"]
    units = child["units_per_layer"]
    
    if len(units) != nl:
        
        if nl == 3:
            new_units = [random.choice([512, 600, 700]), 256, 128]
        elif nl == 4:
            new_units = [random.choice([600, 700, 800]), 400, 200, 100]
        elif nl == 5:
            new_units = [random.choice([700, 800, 900]), 500, 300, 150, 75]
        else:  # 6 layers
            new_units = [random.choice([800, 900, 1024]), 600, 400, 200, 100, 50]
        
        child["units_per_layer"] = new_units
    
    return child

def evaluate_inverse_cfg(cfg, X_tr, y_tr, X_va, y_va):
    """evaluation configuration """
    try:
        
        tf.keras.backend.clear_session()
        
        model = build_inverse_model(X_tr.shape[1], cfg)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)]
        
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0,
            callbacks=cb
        )
        
        val_mse = float(min(hist.history["val_loss"]))
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        
        
        del model
        tf.keras.backend.clear_session()
        
        return val_mse, best_epoch
        
    except Exception as e:
        print(f"error in avaluation configuration : {e}")
        tf.keras.backend.clear_session()
        return float('inf'), 0

# ===========================
# Run GA for INVERSE 
# ===========================
POP_SIZE, N_GEN, ELITE_K, MUT_P = 12, 8, 3, 0.4
population = [random_inverse_config() for _ in range(POP_SIZE)]
hall = []

print("start")
print(f"p: {POP_SIZE}, g: {N_GEN}")

for gen in range(N_GEN):
    scored = []
    for i, cfg in enumerate(population):
        print(f"  eval modle {i+1}/{len(population)} in generation {gen+1}...")
        val_mse, best_epoch = evaluate_inverse_cfg(cfg, X_train_p, y_train_p, X_val_p, y_val_p)
        if val_mse < float('inf'):
            scored.append((val_mse, best_epoch, cfg))
            print(f"    MSE: {val_mse:.6f}, layers: {cfg['n_layers']}, units: {cfg['units_per_layer'][:2]}...")
        else:
            print(f"    ❌ error in eval")
    
    if not scored:
        print("no configuration")
        population = [random_inverse_config() for _ in range(POP_SIZE)]
        continue
        
    scored.sort(key=lambda x: x[0])
    best_gen = scored[0]
    hall.append({"gen": gen, "val_mse": best_gen[0], "best_epoch": best_gen[1], "cfg": best_gen[2]})
    
    print(f"[Gen {gen+1}] best val MSE = {best_gen[0]:.6f} | layers={best_gen[2]['n_layers']} | units={best_gen[2]['units_per_layer']}")
    
    new_pop = [scored[i][2] for i in range(min(ELITE_K, len(scored)))]
    
    while len(new_pop) < POP_SIZE:
        if len(scored) >= 2:
            a, b = random.sample(scored[:min(4, len(scored))], 2)
            child = crossover_inverse(a[2], b[2])
            child = mutate_inverse(child, p=MUT_P)
            new_pop.append(child)
        else:
            new_pop.append(random_inverse_config())
    
    population = new_pop

if hall:
    hall.sort(key=lambda h: h["val_mse"])
    best_cfg = hall[0]["cfg"]
    print("\n🎯 best configuration :")
    print(json.dumps(best_cfg, indent=2))
    
    with open(ARTIFACTS / "best_inverse_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    # ===========================
    # Final Training on best config for INVERSE
    # ===========================
    print("\n🔥 training with the best configuration...")
    best_model = build_inverse_model(X_train_p.shape[1], best_cfg)
    
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=best_cfg["patience"], restore_best_weights=True)]
    
    hist = best_model.fit(
        X_train_p, y_train_p,
        validation_data=(X_val_p, y_val_p),
        epochs=best_cfg["epochs"],
        batch_size=best_cfg["batch_size"],
        verbose=1,
        callbacks=cb
    )
    
    best_model.save(MODEL_DIR / "trained_inverse_model.h5")
    pd.DataFrame(hist.history).to_csv(ARTIFACTS / "inverse_train_history.csv", index=False)
   
    joblib.dump(scaler_X, MODEL_DIR / "inverse_scaler_X.pkl")
    joblib.dump(scaler_y, MODEL_DIR / "inverse_scaler_y.pkl")

    # ===========================
    # Plots for INVERSE
    # ===========================
    def save_tiff_plot(filename, fig):
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=300, format="tiff")
        plt.close(fig)

    
    fig1 = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["loss"], label="Train MSE")
    plt.plot(hist.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training vs Validation MSE (Inverse Model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["mae"], label="Train MAE")
    plt.plot(hist.history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training vs Validation MAE (Inverse Model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_tiff_plot("inverse_mse_mae_train_val.tiff", fig1)

    # ===========================
    # Test Evaluation for INVERSE
    # ===========================
    def load_inverse_test(csv_path: Path, sequence_length=600):
        """upload test dataset"""
        X_test, y_test, _ = load_and_prepare_inverse_data(csv_path, sequence_length)
        
        if len(X_test) == 0:
            print("❌ not enough data!")
            return None, None
            
        X_test_scaled = scaler_X.transform(X_test)
        y_test_original = y_test  
        
        return X_test_scaled, y_test_original

    if TEST_CSV.exists():
        print("test")
        X_test_p, y_test_true = load_inverse_test(TEST_CSV)
        
        if X_test_p is not None:
            y_test_pred_scaled = best_model.predict(X_test_p, verbose=0)
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
            
            
            mse_param1 = mean_squared_error(y_test_true[:, 0], y_test_pred[:, 0])
            mse_param2 = mean_squared_error(y_test_true[:, 1], y_test_pred[:, 1])
            total_mse = (mse_param1 + mse_param2) / 2
            
            
            fig2 = plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(y_test_true[:, 0], y_test_pred[:, 0], alpha=0.7, s=50)
            min_val = min(y_test_true[:, 0].min(), y_test_pred[:, 0].min())
            max_val = max(y_test_true[:, 0].max(), y_test_pred[:, 0].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            plt.xlabel('real value Param1')
            plt.ylabel('predicted value Param1')
            plt.title(f'Param1 - MSE: {mse_param1:.6f}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.scatter(y_test_true[:, 1], y_test_pred[:, 1], alpha=0.7, s=50)
            min_val = min(y_test_true[:, 1].min(), y_test_pred[:, 1].min())
            max_val = max(y_test_true[:, 1].max(), y_test_pred[:, 1].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            plt.xlabel('real value Param2')
            plt.ylabel('predicted value Param2')
            plt.title(f'Param2 - MSE: {mse_param2:.6f}')
            plt.grid(True, alpha=0.3)
            
            save_tiff_plot("inverse_test_scatter.tiff", fig2)
            
            # نمایش نمونه‌ای از نتایج
            print(f"\n sample of test:")
            for i in range(min(5, len(y_test_true))):
                print(f"  sample {i+1}:")
                print(f"    real:    Param1={y_test_true[i, 0]:.3f}, Param2={y_test_true[i, 1]:.3f}")
                print(f"    prediction: Param1={y_test_pred[i, 0]:.3f}, Param2={y_test_pred[i, 1]:.3f}")
                print(f"    error:     Param1={abs(y_test_true[i, 0] - y_test_pred[i, 0]):.3f}, Param2={abs(y_test_true[i, 1] - y_test_pred[i, 1]):.3f}")
    else:
        print("⚠️ no test dataset")
        total_mse = hall[0]["val_mse"]

    # ===========================
    # Final Report JSON for INVERSE
    # ===========================
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "inverse",
        "input_description": "600 consecutive output values",
        "output_description": "2 parameters [param1, param2]",
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(len(y_test_true)) if 'y_test_true' in locals() else 0,
        "input_dimension": int(X_train.shape[1]),
        "output_dimension": 2,
        "preprocessing": {
            "input_scaling": "StandardScaler",
            "output_scaling": "StandardScaler",
            "output_preprocessing": "abs() + clip(0,1)"
        },
        "best_config": best_cfg,
        "best_val_mse": float(min(hist.history["val_loss"])),
        "early_stopped_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "test_mse": float(total_mse) if 'total_mse' in locals() else None,
        "artifacts": {
            "model_h5": str((MODEL_DIR / "trained_inverse_model.h5").resolve()),
            "scaler_X": str((MODEL_DIR / "inverse_scaler_X.pkl").resolve()),
            "scaler_y": str((MODEL_DIR / "inverse_scaler_y.pkl").resolve()),
            "best_config_json": str((ARTIFACTS / "best_inverse_config.json").resolve()),
            "train_history_csv": str((ARTIFACTS / "inverse_train_history.csv").resolve()),
            "mse_plot_tiff": str((PLOTS_DIR / "inverse_mse_mae_train_val.tiff").resolve()),
            "test_plot_tiff": str((PLOTS_DIR / "inverse_test_scatter.tiff").resolve() if TEST_CSV.exists() else "N/A")
        }
    }
    
    with open(BASE_DIR / "inverse_summary_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ complate traning!")
    if 'total_mse' in locals():
        print(f"📊 MSE test: {total_mse:.6f}")
    
    # ===========================
    # Prediction Function for INVERSE
    # ===========================
    def predict_inverse(outputs_600):
        """
        
        
        Parameters:
        outputs_600: list of 600 outputs
        
        Returns:
        predicted parameters [param1, param2]
        """

        processed_input = np.abs(np.array(outputs_600, dtype=float))
        processed_input = np.clip(processed_input, 0, 1)
        
  
        input_scaled = scaler_X.transform([processed_input])
        

        pred_scaled = best_model.predict(input_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(pred_scaled)[0]
        
        return prediction

   
    print("""
predicted_params = predict_inverse(outputs_600)
print(f"predicted parameter: {predicted_params}")
    """)

else:
    print("❌ no configuration !")

print("🎯 finish!")