import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import joblib

# -----------------------
# Helpers
# -----------------------
def safe_load_np(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1 and arr.dtype == object:
        try:
            arr2 = np.vstack(arr)
            return arr2
        except Exception:
            return arr
    return arr

def make_numeric(X):
    X = np.asarray(X)
    if X.ndim == 1:
        try:
            X = np.vstack(X)
        except Exception:
            raise ValueError("combined_features is 1D and elements aren't stackable.")
    if X.ndim != 2:
        raise ValueError("combined_features must be 2D after conversion.")
    cols = []
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        try:
            colf = col.astype(float)
            cols.append(colf)
        except Exception:
            le = LabelEncoder()
            col_enc = le.fit_transform(col.astype(str)).astype(float)
            cols.append(col_enc)
    X_num = np.column_stack(cols)
    return X_num

def smape_np(y_true, y_pred, epsilon=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred) + epsilon) / 2.0
    return 100.0 * np.mean(num / den)

# -----------------------
# 1. Load data
# -----------------------
df = pd.read_csv("dataset/train.csv")
y_full = df["price"].values
text_embeddings = safe_load_np("dataset/text_embeddings_train.npy")
combined_features = safe_load_np("dataset/combined_features.npy")

# Align samples
n_samples = min(text_embeddings.shape[0], combined_features.shape[0], y_full.shape[0])
text_embeddings = text_embeddings[:n_samples]
combined_features = combined_features[:n_samples]
y = y_full[:n_samples].astype(float)

# Ensure numeric
if not np.issubdtype(combined_features.dtype, np.number) or combined_features.ndim != 2:
    combined_features = make_numeric(combined_features)

# -----------------------
# 2. Train/test split
# -----------------------
X_text_train, X_text_test, X_comb_train, X_comb_test, y_train, y_test = train_test_split(
    text_embeddings, combined_features, y, test_size=0.33, random_state=42
)

# -----------------------
# 3. Scale features
# -----------------------
scaler_text = StandardScaler()
scaler_comb = StandardScaler()

X_text_train = scaler_text.fit_transform(X_text_train)
X_text_test = scaler_text.transform(X_text_test)

X_comb_train = scaler_comb.fit_transform(X_comb_train)
X_comb_test = scaler_comb.transform(X_comb_test)

# -----------------------
# 4. Log-transform target
# -----------------------
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# -----------------------
# 5. Keras SMAPE metric
# -----------------------
def smape_keras(y_true, y_pred):
    epsilon = 1e-8
    num = K.abs(y_true - y_pred)
    den = (K.abs(y_true) + K.abs(y_pred) + epsilon) / 2.0
    return 100.0 * K.mean(num / den)

# -----------------------
# 6. Build two-branch model (wider + lower dropout)
# -----------------------
tf.random.set_seed(42)

# Text branch
input_text = Input(shape=(X_text_train.shape[1],), name="text_input")
t = Dense(768, activation='relu', kernel_regularizer=l2(1e-4))(input_text)
t = BatchNormalization()(t)
t = Dropout(0.25)(t)
t = Dense(384, activation='relu', kernel_regularizer=l2(1e-4))(t)
t = BatchNormalization()(t)
t = Dropout(0.15)(t)

# Combined branch
input_comb = Input(shape=(X_comb_train.shape[1],), name="comb_input")
c = Dense(384, activation='relu', kernel_regularizer=l2(1e-4))(input_comb)
c = BatchNormalization()(c)
c = Dropout(0.15)(c)
c = Dense(192, activation='relu', kernel_regularizer=l2(1e-4))(c)
c = BatchNormalization()(c)
c = Dropout(0.1)(c)

# Merge
merged = Concatenate()([t, c])
x = Dense(384, activation='relu', kernel_regularizer=l2(1e-4))(merged)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)
x = Dense(192, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
output = Dense(1, activation='linear', name="price_out")(x)

model = Model(inputs=[input_text, input_comb], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mae', metrics=[smape_keras])
model.summary()

# -----------------------
# 7. Train
# -----------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    [X_text_train, X_comb_train],
    y_train_log,
    validation_data=([X_text_test, X_comb_test], y_test_log),
    epochs=250,
    batch_size=128,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -----------------------
# 8. Evaluate and inverse transform
# -----------------------
y_pred_log = model.predict([X_text_test, X_comb_test]).ravel()
y_pred = np.expm1(y_pred_log)

test_mae = mean_absolute_error(y_test, y_pred)
test_smape = smape_np(y_test, y_pred)
print(f"\n✅ Final Test MAE: {test_mae:.4f}")
print(f"✅ Final Test SMAPE: {test_smape:.2f}")

# -----------------------
# 9. Plot training history
# -----------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('MAE on log(target)')
plt.xlabel('epoch')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['smape_keras'], label='train_smape')
plt.plot(history.history['val_smape_keras'], label='val_smape')
plt.title('SMAPE (log-scale)')
plt.xlabel('epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# 10. Save model & scalers
# -----------------------
model.save("price_prediction_two_branch_log_wide.keras")
np.save("scaler_text_meanvar.npy", np.array([scaler_text.mean_, scaler_text.var_], dtype=object), allow_pickle=True)
np.save("scaler_comb_meanvar.npy", np.array([scaler_comb.mean_, scaler_comb.var_], dtype=object), allow_pickle=True)
joblib.dump(scaler_text, "scaler_text.save")
joblib.dump(scaler_comb, "scaler_comb.save")

print("✅ Model and scalers saved.")
