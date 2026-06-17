import numpy as np
import matplotlib.pyplot as plt

from ml_genn.callbacks import Checkpoint, SpikeRecorder, Callback
from ml_genn.compilers import EventPropCompiler
from ml_genn.optimisers import Adam
from ml_genn.regularisers import SpikeCount
from ml_genn.serialisers import Numpy
import rawSC_util

from time import perf_counter
import os, sys, json

from rawSC_model import create_model

p= {"RECURR": False,
    "LEARN_FF": True,
    "LEARN_REC": False,
    "NUM_INPUT": 80,
    "INPUT_FRAMES": 200,
    "INPUT_FRAME_TIMESTEPS": 5,
    "LR": 1e-3,
    "DELAY_LR": 1e-1,
    "NUM_EPOCHS": 500,
    "NUM_LAYER": 2,
    "NUM_OUTPUT": 35,
    "NUM_HIDDEN": 1024,
    "INPUT_HIDDEN_MEAN": 0.15,
    "INPUT_HIDDEN_SD": 0.05,
    "FF_DELAY_INIT": 50,
    "HIDDEN_HIDDEN_MEAN": 0.02,
    "HIDDEN_HIDDEN_SD": 0.03,
    "RECURR_MEAN": 0.0,
    "RECURR_SD": 0.01,
    "RECURR_DELAY_INIT": 0,
    "HIDDEN_OUT_MEAN": 0.0,
    "HIDDEN_OUT_SD": 0.03,
    "K_REG": [ 5e-12, 5e-12 ],
    "BATCH_SIZE": 256,
    "INPUT_SCALE": 0.03,
    "SEED": 42,
    "NAME": "test0",
    "PLOT_EXAMPLES": False,
    "SHUFFLE": True,
    "SHIFT": 4,
}


class EaseInSchedule(Callback):
    def create_state(self, compiled_network, **kwargs):
        return [s for s in compiled_network.optimiser_state]

    def on_batch_begin(self, state, batch):
        # Set parameter to return value of function
        for s in state:
            if s.alpha < 0.001 :
                s.alpha = (0.001 / 100.0) * (1.05 ** batch)
            else:
                s.alpha = 0.001

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value

print(p)
with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)

np.random.seed(p["SEED"])

basename = "../data/rawSC/"
X_train = np.load(basename+"audio_train.npy")*p["INPUT_SCALE"]
X_train = np.transpose(X_train, [ 0, 2, 1])
Y_train = np.load(basename+"labels_train.npy")
X_val = np.load(basename+"audio_validation.npy")*p["INPUT_SCALE"]
X_val = np.transpose(X_val, [ 0, 2, 1])
Y_val = np.load(basename+"labels_validation.npy")
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
serialiser = Numpy(p["NAME"]+"_checkpoints")
input, network, ff, rec, hidden, output = create_model(p)

max_example_timesteps = p["INPUT_FRAMES"]*p["INPUT_FRAME_TIMESTEPS"]
optimisers = {"all_connections": {"weight": Adam(p["LR"]/100.0)}}
if p["LEARN_FF"]:
    for conn in ff:
        optimisers[conn] = {"delay": Adam(p["DELAY_LR"])}
if p["RECURR"] and p["LEARN_REC"]:
    for conn in rec:
        optimisers[conn] = {"delay": Adam(p["DELAY_LR"])}
regularisers = {} 
for i, hid in enumerate(hidden):
    regularisers[hid] = SpikeCount(strength=p["K_REG"][i], target=14)

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             max_spikes=1500,
                             batch_size=p["BATCH_SIZE"], rng_seed=p["SEED"])

compiled_net = compiler.compile(network, name=p["NAME"], optimisers=optimisers,
                                regularisers=regularisers)

if p["PLOT_EXAMPLES"]:
    fig, ax = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            id = i*10 + j
            ax[i][j].imshow(X_train[id,:,:].T)
            ax[i][j].yaxis.set_inverted(False)


if p["SHIFT"] > 0:
    shift = rawSC_util.Shift(p["SHIFT"], p["NUM_INPUT"])

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()
    callbacks = [EaseInSchedule()] #, SpikeRecorder(input, key="input_spikes")]
    for i, hid in enumerate(hidden):
        callbacks.append(SpikeRecorder(hid, key="hidden_spikes_"+str(i), record_counts=True))
    val_callbacks = []
    best_e, best_acc = 0, 0
    for e in range(p["NUM_EPOCHS"]):
        X = shift(X_train) if p["SHIFT"] > 0 else X_train
        metrics, val_metrics, cb, val_cb  = compiled_net.train_validate({input: X},
                                                {output: Y_train},
                                                start_epoch=e, num_epochs=1, 
                                                shuffle=p["SHUFFLE"], callbacks=callbacks,
                                                validation_callbacks=val_callbacks,
                                                validation_x={input: X_val},
                                                validation_y={output: Y_val},
        )

        if p["PLOT_EXAMPLES"] and e == 0:
            fig, ax = plt.subplots(10,10)
            for i in range(10):
                for j in range(10):
                    id = i*10 + j
                    ax[i][j].scatter(cb["input_spikes"][0][id],cb["input_spikes"][1][id],s=2)
                    ax[i][j].set_xlim([ 0, 1000])
            plt.show()
        f = open(p["NAME"]+"_results.txt","a")
        f.write(f"{e} {np.mean(np.mean(cb['hidden_spikes_0']))} ")
        hidden_spikes = np.zeros(p["NUM_HIDDEN"])
        for cb_d in cb['hidden_spikes_0']:
            hidden_spikes += cb_d
        
        Conn = compiled_net.connection_populations[ff[0]]
        Conn.vars["weight"].pull_from_device()
        g_view = Conn.vars["weight"].view.reshape((p["NUM_INPUT"], p["NUM_HIDDEN"]))
        g_view[:,hidden_spikes==0] += 0.002
        Conn.vars["weight"].push_to_device()
  
        for i in range(1,p["NUM_LAYER"]):
            f.write(f"{np.mean(np.mean(cb[f'hidden_spikes_{i}']))} ")
            hidden_spikes = np.zeros(p["NUM_HIDDEN"])
            for cb_d in cb['hidden_spikes_'+str(i)]:
                hidden_spikes += cb_d
            Conn = compiled_net.connection_populations[ff[i]]
            Conn.vars["weight"].pull_from_device()
            g_view = Conn.vars["weight"].view.reshape((p["NUM_HIDDEN"], p["NUM_HIDDEN"]))
            g_view[:,hidden_spikes==0] += 0.002
            Conn.vars["weight"].push_to_device()
        

        if metrics[output].result > best_acc:
            best_acc = metrics[output].result
            best_e = e
            compiled_net.save((0,),serialiser)

        f.write(f"{metrics[output].result} {val_metrics[output].result}\n")
        f.close()
        print(f"train: {metrics[output].result}, val: {val_metrics[output].result}")
