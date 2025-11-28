# training omniglot with convulutional neural network and (
import os, math, argparse
import numpy as np
import matplotlib.pyplot as plt
import json

# --------------------------- Import ml_genn safely ----------------------------
try:
    import ml_genn
    from ml_genn import InputLayer, Layer, SequentialNetwork
    from ml_genn.callbacks import Checkpoint
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.connectivity import Dense
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
parser.add_argument("--alphabet", type=str, default="0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--example_time", type=float, default=20.0)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--curriculum", action="store_true")
parser.add_argument("--save_examples", action="store_true")
parser.add_argument("--outdir", type=str, default="latency_omniglot")

# --- convolution and augmentation options ---
parser.add_argument("--augment", type=str, default="none",
    choices=["none", "zoom", "zoom_contrast", "shift", "shift_contrast", 
    "shift_zoom","shift_zoom_contrast","rotation","rotation_contrast", 
    "rotation_zoom","rotation_zoom_contrast", "rotation_shift", 
    "rotation_shift_contrast","rotation_shift_zoom","rotation_shift_zoom_contrast",  
    "shear","shear_contrast","shear_zoom_contrast", "shear_zoom", "shear_shift", 
    "shear_shift_contrast", "shear_shift_zoom", "shear_shift_zoom_contrast",
    "rotation_shear", "rotation_shear_contrast", "rotation_shear_zoom",
    "rotation_shear_zoom_contrast", "rotation_shift_shear", "rotation_shift_shear_contrast", 
    "rotation_shift_shear_zoom", "rotation_shift_shear_zoom_contrast"],
   
    help="Type of augmentation applied per epoch.")
parser.add_argument("--dropout", type=float, default=0.0,
    help="Apply dropout regularization with given rate (0 = disabled)")
            
args = parser.parse_args()

SHOW_EXAMPLE = False
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
     "AUGMENTATIONS": args.augment,
     }

p["ALPHABETS"] = parse_alphabet_spec(args.alphabet)

# --------------------------- Save settings ----------------------------
fname = f"{args.outdir}/parameters.json"
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname,"w") as f:
    json.dump(p,f,indent=4)


np.random.seed(p["SEED"])
tf.random.set_seed(p["SEED"])

NUM_INPUT = p["TARGET_H"] * p["TARGET_W"]

# --------------------------- Dataset loading ----------------------------
#X, y, NUM_OUTPUT = load_omniglot(p["ALPHABETS"], p["TARGET_H"], p["TARGET_W"], p["INVERT"])

X, y, alph_ids, char_ids, NUM_OUTPUT = load_omniglot(
       p["ALPHABETS"], p["TARGET_H"], p["TARGET_W"], p["INVERT"])

print("Unique alph_ids:", np.unique(alph_ids))
print("Count:", np.bincount(alph_ids))

# Show grids for selected alphabets
show_alphabet_grids(X, alph_ids, char_ids, select_alphabets=p["ALPHABETS"], save_dir="grids")

for a in p["ALPHABETS"]:
    n_chars = np.unique(char_ids[alph_ids == a]).size
    print(f"[Check] Alphabet {a} has {n_chars} characters")
    
if args.save_examples:
    print("[Data] Saving full alphabet examples ...")
    save_examples(X, alph_ids, char_ids, args.outdir, select_alphabets=p["ALPHABETS"])
       
X_train, y_train, X_val, y_val, train_idx, val_idx = stratified_split(X, y, args.val_fraction)
print(f"[Data] Train={len(X_train)}, Val={len(X_val)}, Classes={NUM_OUTPUT}")


# --------------------------- Network ----------------------------
os.makedirs(args.outdir, exist_ok=True)
#serialiser = Numpy(os.path.join(args.outdir, f"omniglot_alphabet{args.alphabet}"))

network = SequentialNetwork(default_params)
with network:
    inp = InputLayer(SpikeInput(max_spikes=args.batch_size * NUM_INPUT), NUM_INPUT)
    hid = Layer(Dense(Normal(mean=p["IN_HID_MEAN"], sd=p["IN_HID_SD"])),
                LeakyIntegrateFire(v_thresh=p["THRESH"], tau_mem=p["TAU_MEM"]),
                args.hidden, Exponential(p["TAU_SYN"]))
    out = Layer(Dense(Normal(mean=p["HID_OUT_MEAN"], sd=p["HID_OUT_SD"])),
                LeakyIntegrate(tau_mem=p["TAU_MEM_OUT"], readout=p["READOUT"]),
                NUM_OUTPUT, Exponential(p["TAU_SYN"]))

compiler = EventPropCompiler(
    example_timesteps=int(math.ceil(args.example_time / args.dt)),
    losses=p["LOSS"],
    optimiser=Adam(args.lr),
    batch_size=args.batch_size,
    dt=args.dt
).compile(network)


# --------------------------- Training loop ----------------------------
train_accs, val_accs = [], []

with compiler:
    for epoch in range(1, args.epochs+1):
        print(f"\n🌀 Epoch {epoch}/{args.epochs} — applying {args.augment}")
        X_aug = augment_images(X_train, mode=args.augment)
       
        # ---------------------------------------------------------
        # Save augmented grid ONLY for the first epoch (epoch == 1)
        # ---------------------------------------------------------
        if args.save_examples and epoch == 1:
            alphabet_to_plot = p["ALPHABETS"][0]

            save_augmented_grid(
                X_aug=X_aug,
                alph_ids=alph_ids[train_idx],
                char_ids=char_ids[train_idx],
                outdir=os.path.join(args.outdir, "augmented_grids"),
                epoch=epoch,
                alphabet_id=alphabet_to_plot,
                augment_name=args.augment
            )
         
       
        # --- Apply dropout regularization if enabled ---
        if args.dropout > 0:
            print(f"Applying dropout regularization (rate={args.dropout})")
            X_aug = apply_dropout(X_aug, rate=args.dropout)
     
     
   
       
        if SHOW_EXAMPLE:
            plot_examples(X_aug, p["TARGET_H"], p["TARGET_W"], 20, 20)
        train_spikes = linear_latency_encode_data(
            (X_aug * 255).astype(np.uint8),
            p["EXAMPLE_TIME"] - 2*p["DT"], 2*p["DT"]
        )
        val_spikes = linear_latency_encode_data(
            (X_val * 255).astype(np.uint8),
            p["EXAMPLE_TIME"] - 2*p["DT"], 2*p["DT"]
        )
        metrics, val_metrics, _, _ = compiler.train(
            {inp: train_spikes}, {out: y_train},
            num_epochs=1, start_epoch=epoch, shuffle=True,
            validation_x={inp: val_spikes}, validation_y={out: y_val}, callbacks=[], validation_callbacks=[]
        )
        tr_acc = get_accuracy(metrics)
        va_acc = get_accuracy(val_metrics)
        print(f"Epoch {epoch:03d}: train={tr_acc*100:.2f}%  val={va_acc*100:.2f}%")
                  
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# SAVE ACCURACY ARRAYS FOR LATER ANALYSIS
import json
acc_path = os.path.join(args.outdir, f"acc_{args.augment}.json")
with open(acc_path, "w") as f:
    json.dump({"train": train_accs, "val": val_accs}, f)

print(f"[INFO] Saved accuracy data to {acc_path}")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# --------------------------- Plot ----------------------------
plt.figure(figsize=(7,5))
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend(); plt.grid(True)
plt.title(f"Omniglot Training (Alphabet {args.alphabet})\n"
          f"Augmentation: {args.augment} | Hidden={args.hidden}")
fname = f"accuracy_curve_{args.augment}_alphabet{args.alphabet}_hidden{args.hidden}.png"
plt.savefig(os.path.join(args.outdir, fname), dpi=150)
print(f"✅ Plot saved: {fname}")
