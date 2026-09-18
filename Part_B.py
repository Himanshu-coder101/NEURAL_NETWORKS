# ============================================================
# PROGRAMMING ASSIGNMENT #1
# PART B - EFFECT OF LEARNING RATE
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
#    Standardization statistics come ONLY from the complete
#    training set and are reused for validation.
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


Xtr_std, ytr, Xv_std, yv = make_tensors(
    X_train_std,
    y_train,
    X_val_std,
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

    # Same architecture for every learning rate
    model = nn.Linear(100, 1)

    # Required MSE loss
    loss_fn = nn.MSELoss()

    # Required SGD optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = loss_fn(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

        # Evaluate after the epoch's final update
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
# 5. LEARNING-RATE EXPERIMENT
#
# These values are deliberately spread out so that the
# different behaviors are visible:
#
# 0.001 -> slow convergence
# 0.005 -> faster convergence
# 0.01  -> fast and stable convergence
# 0.08  -> large-step / oscillatory behavior
#
# A logarithmic y-axis is used below because MSE spans
# several orders of magnitude.
# ============================================================

learning_rates = [
    0.001,
    0.005,
    0.01,
    0.08
]

val_losses_lr = {}


for lr in learning_rates:

    # Same initialization and same shuffle sequence
    # for every learning-rate experiment.
    torch.manual_seed(42)

    _, train_loss, val_loss = train_model(
        Xtr_std,
        ytr,
        Xv_std,
        yv,
        lr=lr,
        batch_size=32,
        epochs=100
    )

    val_losses_lr[lr] = val_loss


# ============================================================
# 6. PRINT ALL 100 VALIDATION MSE VALUES
# ============================================================

print("\n" + "=" * 85)
print("PART B - VALIDATION MSE FOR DIFFERENT LEARNING RATES")
print("=" * 85)

print(
    f"{'Epoch':>5} | "
    f"{'LR=0.001':>15} | "
    f"{'LR=0.005':>15} | "
    f"{'LR=0.01':>15} | "
    f"{'LR=0.08':>15}"
)

print("-" * 85)

for epoch in range(100):
    print(
        f"{epoch + 1:5d} | "
        f"{val_losses_lr[0.001][epoch]:15.4f} | "
        f"{val_losses_lr[0.005][epoch]:15.4f} | "
        f"{val_losses_lr[0.01][epoch]:15.4f} | "
        f"{val_losses_lr[0.08][epoch]:15.4f}"
    )


# ============================================================
# 7. BEST VALIDATION MSE FOR EACH LEARNING RATE
# ============================================================

print("\n" + "=" * 65)
print("BEST VALIDATION PERFORMANCE")
print("=" * 65)

for lr in learning_rates:

    losses = val_losses_lr[lr]

    best_epoch = min(
        range(len(losses)),
        key=lambda i: losses[i]
    )

    best_mse = losses[best_epoch]

    print(
        f"LR = {lr:<7} | "
        f"Best Val MSE = {best_mse:12.4f} | "
        f"Epoch = {best_epoch + 1}"
    )


# ============================================================
# 8. PLOT VALIDATION MSE
#
# Log scale is used on the y-axis so that the slow and fast
# learning-rate curves do not disappear because of scale.
# ============================================================

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    plt.plot(
        range(1, 101),
        val_losses_lr[lr],
        label=f"LR = {lr}"
    )

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("Validation MSE (log scale)")
plt.title("Part B: Effect of Learning Rate")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 9. NOTE
# ============================================================

print("\nRecommended learning rate for subsequent experiments:")
print("0.01 is a sensible choice based on the observed trade-off")
print("between convergence speed and stable validation performance.")
