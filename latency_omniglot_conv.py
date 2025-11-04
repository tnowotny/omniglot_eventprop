#!/usr/bin/env python3
# latency_omniglot_correct_v8_plot_3_l_aug_shift_epoch.py
# Version: with epoch-wise translation-only augmentation + multi-alphabet + curriculum
# Author: Lyazzat Atymtayeva (modified by ChatGPT), 2025

import os, math, argparse, numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import json

PLOT_EXAMPLES = False
SHOW_RASTER = False
DEBUG = False

# --------------------------- Safe import ----------------------------
try:
    import ml_genn
    from ml_genn import InputLayer, Layer, SequentialNetwork
    from ml_genn.callbacks import Checkpoint, SpikeRecorder
    from ml_genn.compilers import EventPropCompiler, InferenceCompiler
    from ml_genn.connectivity import Dense, Conv2D
    from ml_genn.initializers import Normal
    from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
    from ml_genn.optimisers import Adam
    from ml_genn.serialisers import Numpy
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import linear_latency_encode_data
    from ml_genn.compilers.event_prop_compiler import default_params
    print(f"[ml_genn] Detected version: {getattr(ml_genn, '__version__', 'unknown')}")
except Exception as e:
    raise RuntimeError(f"ml_genn not found or broken: {e}")

from utils import *

# --------------------------- Args ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--alphabet", type=str, default="0", help="Alphabet number or range (e.g. '0-2' or '0,1,5')")
parser.add_argument("--invert", action="store_true")
parser.add_argument("--no_minmax", action="store_true")
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--hidden", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--example_time", type=float, default=40.0)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--outdir", type=str, default="latency_omniglot_conv_runs")
parser.add_argument("--curriculum", action="store_true",
                    help="Enable curriculum learning: reuse weights from previous alphabets")
parser.add_argument("--save_examples", action="store_true",
                    help="Save Omniglot examples (grids and individual PNGs) for inspection")
parser.add_argument(
    "--conv", type=str, default="sobel_x",
    choices=["none", "gaussian3", "sharpen", "sobel_x", "sobel_y", "laplace"],
    help="Initial convolution kernel type for input to hidden convolutions"
)
args = parser.parse_args()
print(f"[conv] Kernel: {args.conv!r}")

# --- collect all options in a dictionary
p = {"INVERT": True,
     "SEED": 42,
     "TARGET_H": 28,
     "TARGET_W": 28,
     "TAU_SYN": 5.0,
     "TAU_MEM": 20.0,
     "TAU_MEM_OUT": 20.0,
     "THRESH": 1.0,
     "IN_HID_MEAN": 0.08,
     "IN_HID_SD": 0.2,
     "N_HIDDEN": args.hidden,
     "HID_OUT_MEAN": 0.0,
     "HID_OUT_SD": 0.1,
     "EXAMPLE_TIME": args.example_time,
     "DT": args.dt,
     "LR": args.lr,
     "BATCH_SIZE": args.batch_size,
     "LOSS": "sparse_categorical_crossentropy",
     "READOUT": "avg_var",
     "EPOCHS": args.epochs,
     "CONV_KERNEL": args.conv
     }

p["ALPHABETS"] = parse_alphabet_spec(args.alphabet)

# --------------------------- Save settings ----------------------------
fname = f"{args.outdir}/parameters.json"
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname,"w") as f:
    json.dump(p,f,indent=4)

np.random.seed(args.seed)
tf.random.set_seed(args.seed)


NUM_INPUT = p["TARGET_H"] * p["TARGET_W"]

# --------------------------- Load data ----------------------------
X, y, NUM_OUTPUT = load_omniglot(p["ALPHABETS"], p["TARGET_H"], p["TARGET_W"], p["INVERT"])
X_train, y_train, X_val, y_val = stratified_split(X, y, args.val_fraction)
print(f"[Split:stratified] Train={len(X_train)}  Val={len(X_val)}, Classes={NUM_OUTPUT}")

if args.save_examples:
    save_examples(args.outdir,args.alphabet)
    
# --------------------------- Network ----------------------------
os.makedirs(args.outdir, exist_ok=True)
serialiser = Numpy(os.path.join(args.outdir, f"omniglot_alphabet{args.alphabet}"))
kernel = get_kernel(args.conv)

init_kernels = []
num_kernels = 32
kernel_noise = 0.1
conv_ht = kernel.shape[0]
conv_wd = kernel.shape[1]
init_kernels = np.zeros((conv_ht, conv_wd, 1, num_kernels))
for i in range(num_kernels):
    init_kernels[:,:,0,i]= kernel+kernel_noise*np.random.uniform(size=kernel.shape)
init_kernels = np.asarray(init_kernels)
    
network = SequentialNetwork(default_params)
with network:
    inp = InputLayer(SpikeInput(max_spikes=args.batch_size * NUM_INPUT), (p["TARGET_W"], p["TARGET_H"], 1))
    hid = Layer(Conv2D(init_kernels*2, num_kernels, (conv_ht, conv_wd), flatten=True),
                LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                synapse=Exponential(5.0),record_spikes= True if DEBUG else False)
    out = Layer(Dense(Normal(mean=0.0, sd=0.1)),
                LeakyIntegrate(tau_mem=20.0, readout="avg_var"),
                NUM_OUTPUT, Exponential(5.0))

# --------------------------- Compilers ----------------------------
    train_comp = EventPropCompiler(
        example_timesteps=int(math.ceil(args.example_time/args.dt)),
        losses="sparse_categorical_crossentropy",
        optimiser=Adam(args.lr),
        batch_size=args.batch_size,
        dt=args.dt
    ).compile(network)


# --------------------------- Training loop with best weight saving -----------------
train_accs, val_accs = [], []
best_val_acc = 0.0  # store best validation accuracy
if DEBUG:
    callbacks = [ SpikeRecorder(hid, key="shid", record_counts=True) ]
else:
    callbacks = []
with train_comp:
    for epoch in range(1, args.epochs + 1):
        print(f"\n🌀 Epoch {epoch}/{args.epochs} ...")
        
        # --- Apply small random translations each epoch ---
        #X_shift = shift_images(X_train, max_shift=3)
        #print(X_shift.shape)
        #for i in range(len(X_shift)):
        #    plt.figure()
        #    plt.imshow(X_shift[i,:].reshape((28,28)))
        #    plt.show()
        X_shift = X_train
        X_shift_255 = (X_shift * 255).astype(np.uint8)
        train_spikes = linear_latency_encode_data(X_shift_255, p["EXAMPLE_TIME"] - 2*p["DT"], 2*p["DT"])

        val_255 = (X_val * 255).astype(np.uint8)
        val_spikes = linear_latency_encode_data(val_255, p["EXAMPLE_TIME"] - 2*p["DT"], 2*p["DT"])

        # --- Train one epoch ---
        metrics, val_metrics, cb_data, val_cb_data = train_comp.train(
            {inp: train_spikes}, {out: y_train},
            num_epochs=1, start_epoch=epoch, shuffle=True,
            validation_x={inp: val_spikes}, validation_y={out: y_val}, callbacks=callbacks,
            validation_callbacks=callbacks
        )

        # --- Extract accuracies ---
        train_acc = get_accuracy(metrics[out])
        val_acc = get_accuracy(val_metrics[out])
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if DEBUG:
            n = cb_data["shid"]
            print(f"n_hid= {np.mean(n,axis=1)} +/- {np.std(n,axis=1)}")
            n = val_cb_data["shid"]
            print(f"val_n_hid= {np.mean(n,axis=1)} +/- {np.std(n,axis=1)}")
        print(f"Epoch {epoch:03d}: train={train_acc*100:.2f}%  val={val_acc*100:.2f}%")

        # --- Save weights if validation accuracy improves ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            train_comp.save((0,), serialiser)

plt.figure(figsize=(7, 5))

# Prepare data arrays
epochs = np.arange(1, len(train_accs) + 1)
train_arr = np.array(train_accs) * 100
val_arr = np.array(val_accs) * 100

# --- Plot raw curves (semi-transparent, to see original fluctuation) ---
plt.plot(epochs, train_arr, alpha=0.25, color="tab:blue", label="Train (raw)")
plt.plot(epochs, val_arr, alpha=0.25, color="tab:orange", label="Validation (raw)")

# --- Compute and plot smoothed curves using moving average ---
window = 15  # number of epochs for averaging
smooth_train = moving_average(train_arr, window)
smooth_val = moving_average(val_arr, window)
smooth_epochs = np.arange(1, len(smooth_train) + 1)

plt.plot(smooth_epochs, smooth_train, color="tab:blue", linewidth=2.0,
         label=f"Train (smoothed, w={window})")
plt.plot(smooth_epochs, smooth_val, color="tab:orange", linewidth=2.0,
         label=f"Validation (smoothed, w={window})")

# --- Find and mark the epoch with the highest validation accuracy ---
best_epoch = np.argmax(val_arr) + 1       # +1 because epochs start from 1
best_val = val_arr[best_epoch - 1]
plt.scatter(best_epoch, best_val, color="red", s=60, edgecolor="black", zorder=5,
            label=f"Best val ({best_val:.2f}% @ epoch {best_epoch})")

# --- Labels and styling ---
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"Omniglot Training (Alphabet {args.alphabet}) \n"
          "(Smoothed curves + best validation point)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Save plot to file ---
#plot_path = os.path.join(args.outdir, "accuracy_curve_shift_best.png")

# Save plot with alphabet-specific filename
plot_path = os.path.join(args.outdir,
                         f"accuracy_curve_shift_best_alphabet_{str(args.alphabet).replace(',', '_').replace('-', '_')}.png")
plt.savefig(plot_path, dpi=150)
print(f"\n✅ Smoothed plot with best point saved to: {plot_path}")
