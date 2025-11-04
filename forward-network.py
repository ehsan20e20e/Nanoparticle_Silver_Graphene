#The code for the research presented in the paper titled "A_deep_learning_method_for_extinction_spectrum_prediction_and_graphene-coated_silver_nanoparticles_inverse_design

#This code corresponds to the article's forward Deep Neural Network (DNN) section.
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
BASE_DIR = Path("./nn_regression_ga_outputs")
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
# Data Loading & Correction
# ===========================
def load_dataframe(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :3].copy()
    y = df.iloc[:, -1].astype(float).copy()
    return X, y, df

def correct_targets(y: pd.Series):
    y_corr = y.copy()
    y_corr[y_corr < 0] = -y_corr[y_corr < 0]
    y_corr = np.minimum(y_corr, 1.0)
    return y_corr

X_all, y_raw, _ = load_dataframe(TRAIN_CSV)
y_all = correct_targets(y_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X_all.values, y_all.values, test_size=0.10, random_state=42, shuffle=True
)

# ===========================
 
# ===========================
def preprocess_inputs(X_train, X_val, cfg):
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

# ===========================
# Model Factory
# ===========================
def build_model(input_dim, cfg):
    model = tf.keras.Sequential()
    n_layers = cfg["n_layers"]
    units_list = cfg["units_per_layer"]
    
    
    if len(units_list) != n_layers:
        units_list = [64] * n_layers
    
    for i in range(n_layers):
        units = units_list[i]
        act = cfg["activation"]
        l2 = regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
        model.add(layers.Dense(units, activation=act, kernel_regularizer=l2))
        if cfg["dropout"] > 0:
            model.add(layers.Dropout(cfg["dropout"]))
    model.add(layers.Dense(1, activation="linear"))
    opt = tf.keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss="mse", metrics=["mse", "mae"])
    return model

# ===========================
# ===========================
ACTIVATIONS = ["relu", "tanh"] 

def random_config():
    n_layers = random.randint(1, 6)
    units = [random.choice([32, 64, 128, 256, 512, 1024]) for _ in range(n_layers)]
    return {
        "n_layers": n_layers,
        "units_per_layer": units,
        "activation": random.choice(ACTIVATIONS),
        "dropout": round(random.uniform(0.0, 0.5), 2),
        "l2": round(random.choice([0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]), 6),
        "lr": 10**random.uniform(-5, -2),
        "batch_size": random.choice([128, 256, 512, 1024, 2048]),
        "epochs": 400,
        "patience": 40,
        "c_encoding": random.choice(["numeric", "onehot"]),
    }

def mutate(cfg, p=0.3):
    new_cfg = json.loads(json.dumps(cfg))
    if random.random() < p:
        new_cfg["activation"] = random.choice(ACTIVATIONS)
    if random.random() < p:
        new_cfg["dropout"] = round(np.clip(new_cfg["dropout"] + random.uniform(-0.1, 0.1), 0, 0.5), 2)
    if random.random() < p:
        new_cfg["l2"] = round(random.choice([0.0, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3]), 6)
    if random.random() < p:
        new_cfg["lr"] = 10**random.uniform(-5, -2)
    if random.random() < p:
        new_cfg["batch_size"] = random.choice([128, 256, 512, 1024, 2048])
    if random.random() < p:
        nl = random.randint(1, 6)
        new_cfg["n_layers"] = nl
        new_cfg["units_per_layer"] = [random.choice([32, 64, 128, 256, 512, 1024]) for _ in range(nl)]
    if random.random() < p:
        new_cfg["c_encoding"] = random.choice(["numeric", "onehot"])
    
    
    nl = new_cfg["n_layers"]
    units = new_cfg["units_per_layer"]
    if len(units) != nl:
        if len(units) > nl:
            units = units[:nl]
        else:
            needed = nl - len(units)
            last_unit = units[-1] if units else 64
            units.extend([last_unit] * needed)
        new_cfg["units_per_layer"] = units
    
    return new_cfg

def crossover(a, b):
    child = {}
    for k in a.keys():
        child[k] = random.choice([a[k], b[k]])
    
    
    nl = child["n_layers"]
    units = child["units_per_layer"]
    
    if len(units) > nl:
        units = units[:nl]
    elif len(units) < nl:
        needed = nl - len(units)
        last_unit = units[-1] if units else 64
        units.extend([last_unit] * needed)
    
    child["units_per_layer"] = units
    return child

def evaluate_cfg(cfg, X_tr, y_tr, X_va, y_va):
    try:
        X_tr_p, X_va_p, _ = preprocess_inputs(X_tr, X_va, cfg)
        model = build_model(X_tr_p.shape[1], cfg)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)]
        hist = model.fit(
            X_tr_p, y_tr,
            validation_data=(X_va_p, y_va),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0,
            callbacks=cb
        )
        val_mse = float(min(hist.history["val_loss"]))
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        return val_mse, best_epoch
    except Exception as e:
        print(f"error in configuration: {e}")
        return float('inf'), 0

# ===========================
# Run GA
# ===========================
POP_SIZE, N_GEN, ELITE_K, MUT_P = 24, 12, 5, 0.5
population = [random_config() for _ in range(POP_SIZE)]
hall = []

for gen in range(N_GEN):
    scored = []
    for cfg in population:
        val_mse, best_epoch = evaluate_cfg(cfg, X_train, y_train, X_val, y_val)
        if val_mse < float('inf'):
            scored.append((val_mse, best_epoch, cfg))
    
    if not scored:
        print("no new configuration")
        population = [random_config() for _ in range(POP_SIZE)]
        continue
        
    scored.sort(key=lambda x: x[0])
    best_gen = scored[0]
    hall.append({"gen": gen, "val_mse": best_gen[0], "best_epoch": best_gen[1], "cfg": best_gen[2]})
    print(f"[Gen {gen}] best val MSE = {best_gen[0]:.6f} | encoding={best_gen[2]['c_encoding']}")
    new_pop = [scored[i][2] for i in range(min(ELITE_K, len(scored)))]
    
    while len(new_pop) < POP_SIZE:
        if len(scored) >= 2:
            a, b = random.sample(scored[:min(6, len(scored))], 2)
            child = crossover(a[2], b[2])
            child = mutate(child, p=MUT_P)
            new_pop.append(child)
        else:
            new_pop.append(random_config())
    
    population = new_pop

if hall:
    hall.sort(key=lambda h: h["val_mse"])
    best_cfg = hall[0]["cfg"]
    print("\nBest config:", json.dumps(best_cfg, indent=2))
    with open(ARTIFACTS / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    # ===========================
    # Final Training on best config
    # ===========================
    X_train_p, X_val_p, preprocessors = preprocess_inputs(X_train, X_val, best_cfg)
    best_model = build_model(X_train_p.shape[1], best_cfg)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=best_cfg["patience"], restore_best_weights=True)]
    hist = best_model.fit(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=best_cfg["epochs"],
        batch_size=best_cfg["batch_size"],
        verbose=0,
        callbacks=cb
    )
    best_model.save(MODEL_DIR / "trained_model.h5")
    pd.DataFrame(hist.history).to_csv(ARTIFACTS / "train_history.csv", index=False)

    # ===========================
    # Plots
    # ===========================
    def save_tiff_plot(filename, fig):
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=600, format="tiff")
        plt.close(fig)

    fig1 = plt.figure()
    plt.plot(hist.history["loss"], label="Train MSE")
    plt.plot(hist.history["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training vs Validation MSE")
    plt.legend()
    save_tiff_plot("mse_train_val.tiff", fig1)

    # ===========================
    # Test Evaluation
    # ===========================
    def load_test(csv_path: Path, cfg, preprocessors):
        X_t, y_t, _ = load_dataframe(csv_path)
        y_t_corr = correct_targets(y_t)
        if cfg["c_encoding"] == "numeric":
            scaler = preprocessors["scaler"]
            X_t_p = scaler.transform(X_t.values)
        else:
            scaler, enc = preprocessors["scaler"], preprocessors["encoder"]
            C_te = enc.transform(X_t.values[:, 2].reshape(-1,1))
            AB_te = scaler.transform(X_t.values[:, :2])
            X_t_p = np.hstack([AB_te, C_te])
        return X_t_p, y_t_corr.values

    X_test_p, y_test_corr = load_test(TEST_CSV, best_cfg, preprocessors)
    y_pred = best_model.predict(X_test_p, verbose=0).ravel()

    fig2 = plt.figure()
    plt.plot(y_test_corr, label="True", color="black", linewidth=1.5)
    plt.plot(y_pred, label="Predicted", color="red", linewidth=1.5)
    plt.scatter(range(len(y_test_corr)), y_test_corr, s=14, c="black", marker="o")
    plt.scatter(range(len(y_pred)), y_pred, s=14, c="red", marker="x")
    plt.xlabel("Sample Index")
    plt.ylabel("D")
    plt.title("Test Set: True vs Predicted (continuous + points)")
    plt.legend()
    save_tiff_plot("test_true_vs_pred_continuous.tiff", fig2)

    test_mse = mean_squared_error(y_test_corr, y_pred)

    # ===========================
    # Final Report JSON
    # ===========================
    report = {
        "timestamp": datetime.now().isoformat(),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(len(y_test_corr)),
        "input_columns": ["A","B","C"],
        "target_column": "D",
        "label_corrections": {
            "negatives_reflected_train": int((y_raw < 0).sum()),
            "clipped_above_1_train": int((y_raw > 1).sum()),
            "negatives_reflected_test": int((y_test_corr<0).sum()),
            "clipped_above_1_test": int((y_test_corr>1).sum())
        },
        "best_config": best_cfg,
        "best_val_mse": float(min(hist.history["val_loss"])),
        "early_stopped_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "test_mse": float(test_mse),
        "artifacts": {
            "model_h5": str((MODEL_DIR / "trained_model.h5").resolve()),
            "best_config_json": str((ARTIFACTS / "best_config.json").resolve()),
            "train_history_csv": str((ARTIFACTS / "train_history.csv").resolve()),
            "mse_plot_tiff": str((PLOTS_DIR / "mse_train_val.tiff").resolve()),
            "test_plot_tiff": str((PLOTS_DIR / "test_true_vs_pred_continuous.tiff").resolve())
        }
    }
    with open(BASE_DIR / "summary_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Training finished. Test MSE =", test_mse)
else:
    print("❌ no configuration !")

print("🎯 finish!")