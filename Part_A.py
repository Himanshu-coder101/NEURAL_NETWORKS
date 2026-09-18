# ============================================================
# PROGRAMMING ASSIGNMENT #1
# PART A - FEATURE STANDARDIZATION
# ============================================================

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")

X_train = train.drop(columns=["electricity_consumption_kwh"])
y_train = train["electricity_consumption_kwh"]

X_val = validation.drop(columns=["electricity_consumption_kwh"])
y_val = validation["electricity_consumption_kwh"]


# ============================================================
# 2. STANDARDIZATION
#    Mean and standard deviation are calculated ONLY from
#    the complete training set.
# ============================================================

mean = X_train.mean(axis=0)
std = X_train.std(axis=0, ddof=0)

X_train_std = (X_train - mean) / std
X_val_std = (X_val - mean) / std


# ============================================================
# 3. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

def make_tensors(X_train, y_train, X_val, y_val):
    X_train_tensor = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    X_val_tensor = torch.tensor(
        X_val.values,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    return (
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor
    )


# Standardized data
Xtr_std, ytr, Xv_std, yv = make_tensors(
    X_train_std,
    y_train,
    X_val_std,
    y_val
)

# Raw data
Xtr_raw, _, Xv_raw, _ = make_tensors(
    X_train,
    y_train,
    X_val,
    y_val
)


# ============================================================
# 4. TRAINING FUNCTION
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    lr,
    batch_size=32,
    epochs=100
):
    train_dataset = TensorDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Single linear neuron: 100 inputs -> 1 output
    model = nn.Linear(100, 1)

    # Required loss function
    loss_fn = nn.MSELoss()

    # Required optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        # Mini-batch SGD updates
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_fn(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

        # Evaluate using the final weights of this epoch
        model.eval()

        with torch.no_grad():

            train_predictions = model(X_train)
            train_mse = loss_fn(
                train_predictions,
                y_train
            ).item()

            val_predictions = model(X_val)
            val_mse = loss_fn(
                val_predictions,
                y_val
            ).item()

        train_losses.append(train_mse)
        val_losses.append(val_mse)

    return model, train_losses, val_losses


# ============================================================
# 5. EXPERIMENT 1 - STANDARDIZED FEATURES
# ============================================================

torch.manual_seed(42)

model_std, train_std_loss, val_std_loss = train_model(
    Xtr_std,
    ytr,
    Xv_std,
    yv,
    lr=0.01,
    batch_size=32,
    epochs=100
)


# ============================================================
# 6. EXPERIMENT 2 - RAW FEATURES
#
# lr=0.001 is used because lr=0.01 was unstable for the
# raw feature scale.
# ============================================================

torch.manual_seed(42)

model_raw, train_raw_loss, val_raw_loss = train_model(
    Xtr_raw,
    ytr,
    Xv_raw,
    yv,
    lr=0.001,
    batch_size=32,
    epochs=100
)


# ============================================================
# 7. PRINT ALL 100 EPOCHS
# ============================================================

print("\n" + "=" * 80)
print("PART A - MSE FOR EACH EPOCH")
print("=" * 80)

print(
    f"{'Epoch':>5} | "
    f"{'Raw Train':>15} | "
    f"{'Raw Val':>15} | "
    f"{'Std Train':>15} | "
    f"{'Std Val':>15}"
)

print("-" * 80)

for epoch in range(100):
    print(
        f"{epoch + 1:5d} | "
        f"{train_raw_loss[epoch]:15.4f} | "
        f"{val_raw_loss[epoch]:15.4f} | "
        f"{train_std_loss[epoch]:15.4f} | "
        f"{val_std_loss[epoch]:15.4f}"
    )


# ============================================================
# 8. PLOT RESULTS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)


# Raw features
axes[0].plot(
    range(1, 101),
    train_raw_loss,
    label="Training MSE"
)

axes[0].plot(
    range(1, 101),
    val_raw_loss,
    label="Validation MSE"
)

axes[0].set_title("Raw Features")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].legend()
axes[0].grid(True)


# Standardized features
axes[1].plot(
    range(1, 101),
    train_std_loss,
    label="Training MSE"
)

axes[1].plot(
    range(1, 101),
    val_std_loss,
    label="Validation MSE"
)

axes[1].set_title("Standardized Features")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")
axes[1].legend()
axes[1].grid(True)


fig.suptitle(
    "Part A: Raw vs Standardized Features",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ============================================================
# 9. FINAL MSE
# ============================================================

print("\n" + "=" * 55)
print("FINAL MSE AFTER 100 EPOCHS")
print("=" * 55)

print(
    f"Raw Training MSE           : {train_raw_loss[-1]:.4f}"
)
print(
    f"Raw Validation MSE         : {val_raw_loss[-1]:.4f}"
)
print(
    f"Standardized Training MSE  : {train_std_loss[-1]:.4f}"
)
print(
    f"Standardized Validation MSE: {val_std_loss[-1]:.4f}"
)
