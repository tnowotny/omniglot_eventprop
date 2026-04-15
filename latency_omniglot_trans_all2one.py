
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
    from ml_genn.callbacks import SpikeRecorder, ConnVarRecorder
    from ml_genn.connectivity import Dense
    from ml_genn.initializers import Normal
    from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
    from ml_genn.optimisers import Adam
    from ml_genn.serialisers import Numpy
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import linear_latency_encode_data
    from ml_genn.compilers.event_prop_compiler import default_params
    from ml_genn.serialisers import Numpy as NumpySerialiser
    from ml_genn.compilers import InferenceCompiler
    print(f"[ml_genn] Detected version: {getattr(ml_genn, '__version__', 'unknown')}")
except Exception as e:
    raise RuntimeError(f"ml_genn not found or broken: {e}")

from utils_tn import *

# --------------------------- Args ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet", type=str, default="0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--example_time", type=float, default=20.0)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--outdir", type=str, default="latency_omniglot_a2o")
parser.add_argument("--init_from", type=str, default="",
    help="Path to folder with weights_input_hidden.npy and weights_hidden_output.npy")
parser.add_argument("--load_head", action="store_true",
    help="Also load Hidden->Output weights (requires same output shape).")
parser.add_argument("--dropout", type=float, default=0.0,
    help="Apply dropout regularization with given rate (0 = disabled)")
parser.add_argument("--receptive_fields", type=int, default=1,
                    help="Whether to (1) plot receptive fields or (0) not.")
args = parser.parse_args()

SHOW_EXAMPLE = False
# --- collect all options in a dictionary
p = {"SEED": 42,
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
     "T_TRIAL": args.example_time,
     "DT": args.dt,
     "LR": args.lr,
     "BATCH_SIZE": args.batch_size,
     "LOSS": "sparse_categorical_crossentropy",
     "READOUT": "avg_var_exp_weight", #was avg_var
     "EPOCHS": args.epochs,
     "AUGMENTATIONS": [ "rotate", "shear", "zoom", "shift" ], 
     #"AUGMENTATIONS": [ "rotate" ], 
     "VAL_FRACTION": args.val_fraction,
}

# --------------------------- Save settings ----------------------------
fname = f"{args.outdir}/parameters.json"
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname,"w") as f:
    json.dump(p,f,indent=4)

np.random.seed(p["SEED"])
tf.random.set_seed(p["SEED"])

NUM_INPUT = p["TARGET_H"] * p["TARGET_W"]

X, y, alph_ids, char_ids, remap, NUM_OUTPUT = load_omniglot(split= "train")
if SHOW_EXAMPLE:
    show_example(X)
X_train, X_val, y_train, y_val = validation_split(X, y, p["VAL_FRACTION"])
X_train = rescale_images(X_train, p["TARGET_H"],p["TARGET_W"],DEBUG=False)
X_val = rescale_images(X_val, p["TARGET_H"],p["TARGET_W"],DEBUG=False)
if SHOW_EXAMPLE:
    show_example(X_val)
flat_size = X_val.shape[1]*X_val.shape[2]
X_val = X_val.reshape(-1, flat_size)
val_spikes = linear_latency_encode_data(
    X_val, p["T_TRIAL"] - 2 * p["DT"], 2 * p["DT"]
)  

for a in np.unique(alph_ids):
    n_chars = np.unique(char_ids[alph_ids == a]).size
    print(f"[Check] Alphabet {a} has {n_chars} characters")

# === Directory for storing trained weights (checkpoint) ===
checkpoint_dir = os.path.join(args.outdir,"checkpoints")
os.makedirs(args.outdir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
serialiser = NumpySerialiser(checkpoint_dir)

# --------------------------- Network ----------------------------
network = SequentialNetwork(default_params)
with network:
    inp = InputLayer(SpikeInput(max_spikes=p["BATCH_SIZE"] * NUM_INPUT), NUM_INPUT)
    hid = Layer(Dense(Normal(mean=p["IN_HID_MEAN"], sd=p["IN_HID_SD"])),
                LeakyIntegrateFire(v_thresh=p["THRESH"], tau_mem=p["TAU_MEM"]),
                args.hidden, Exponential(p["TAU_SYN"]),
                record_spikes=True)
    out = Layer(Dense(Normal(mean=p["HID_OUT_MEAN"], sd=p["HID_OUT_SD"])),
                LeakyIntegrate(tau_mem=p["TAU_MEM_OUT"], readout=p["READOUT"]),
                NUM_OUTPUT, Exponential(p["TAU_SYN"]), record_spikes=True)

#hid_pop = hid.population()                      
#i2h_conn = hid_pop.incoming_connections[0] 

compiled_net = EventPropCompiler(
    example_timesteps=int(math.ceil(p["T_TRIAL"] / p["DT"])),
    losses=p["LOSS"],
    optimiser=Adam(p["LR"]),
    batch_size=p["BATCH_SIZE"],
    dt=p["DT"]
).compile(network)


# --------------------------- Training loop ----------------------------
train_accs, val_accs = [], []
train_sn, val_sn = [], []
with compiled_net:
    
    if args.init_from:
        load_genn_weights(
            compiled_net, hid, out, args.init_from,
            load_input_hidden=True,
            load_hidden_output=False,
            strict_shapes=True
        )
        
    # === callbacks ===
    callbacks = [
        SpikeRecorder(hid, "hid", record_counts=True),
        Checkpoint(serialiser)  
    ]
    val_callbacks = [
        SpikeRecorder(hid, "hid", record_counts=True)
    ]

    epoch = 0
    resfile= open(os.path.join(args.outdir,"results.txt"), "w")
    resfile.write(f"# train_spikes val_spikes train_acc val_acc\n")
    resfile.close()
    for epoch in range(p["EPOCHS"]):
        X_aug = augment_images(X_train, augmentations=p["AUGMENTATIONS"])
        if SHOW_EXAMPLE:
            show_example(X_aug)
        flat_size = X_aug.shape[1]*X_aug.shape[2]
        X_aug = X_aug.reshape(-1, flat_size)

        # Encode to spikes
        train_spikes = linear_latency_encode_data(
            X_aug, p["T_TRIAL"] - 2 * p["DT"], 2 * p["DT"]
        )

        # One epoch step
        metrics, val_metrics, cb_data, val_cb_data = compiled_net.train(
            {inp: train_spikes}, {out: y_train},
            num_epochs=1, start_epoch=epoch, shuffle=True,
            validation_x={inp: val_spikes}, validation_y={out: y_val},
            callbacks=callbacks, validation_callbacks=val_callbacks
        )
          
        tr_acc = metrics[out].result
        va_acc = val_metrics[out].result
        
        print(f"Epoch {epoch:03d} | train={tr_acc*100:.2f}%  val={va_acc*100:.2f}%")
        print(f"Hidden spikes training: {np.mean(cb_data['hid'])} and validation: {np.mean(val_cb_data['hid'])}")
        resfile= open(os.path.join(args.outdir,"results.txt"), "a")
        resfile.write(f"{np.mean(cb_data['hid'])} {np.mean(val_cb_data['hid'])} {tr_acc} {va_acc}\n")
        resfile.close()

