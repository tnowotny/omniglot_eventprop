from ml_genn import Connection, Network, Population
from ml_genn.connectivity import Dense
from ml_genn.neurons import LeakyIntegrateFire, LeakyIntegrateFireInput, LeakyIntegrate
from ml_genn.synapses import Exponential
from ml_genn.initializers import Normal, Uniform

def create_model(p):
    network = Network()
    with network:
        # Populations
        input = Population(LeakyIntegrateFireInput(v_thresh=1.0,tau_mem=20.0,input_frames=p["INPUT_FRAMES"],input_frame_timesteps=p["INPUT_FRAME_TIMESTEPS"]),
                           p["NUM_INPUT"], record_spikes=True)
        hidden = [Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                             p["NUM_HIDDEN"], record_spikes=True)]
        for i in range(p["NUM_LAYER"]-1):
            hidden.append(Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                            p["NUM_HIDDEN"], record_spikes=True))
        output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                            p["NUM_OUTPUT"],name="output")

        # Connections
        ff = [Connection(input, hidden[0], Dense(Normal(mean=p["INPUT_HIDDEN_MEAN"], sd=p["INPUT_HIDDEN_SD"]), Uniform(0, p["FF_DELAY_INIT"])),
                         Exponential(5.0), max_delay_steps=1000)]
        if p["RECURR"]:
            rec = [Connection(hidden[0], hidden[0], Dense(Normal(mean=p["RECURR_MEAN"], sd=p["RECURR_SD"]), Uniform(0, p["RECURR_DELAY_INIT"])),
                              Exponential(5.0), max_delay_steps=1000)]
        else:
            rec = [None]
        for i in range(p["NUM_LAYER"]-1):
            ff.append(Connection(hidden[i], hidden[i+1], Dense(Normal(mean=p["HIDDEN_HIDDEN_MEAN"], sd=p["HIDDEN_HIDDEN_SD"]), Uniform(0, p["FF_DELAY_INIT"])),
                            Exponential(5.0), max_delay_steps=1000))
            if p["RECURR"]:
                rec.append(Connection(hidden[i+1], hidden[i+1], Dense(Normal(mean=p["RECURR_MEAN"], sd=p["RECURR_SD"]), Uniform(0, p["RECURR_DELAY_INIT"])),
                                      Exponential(5.0), max_delay_steps=1000))
            else:
                rec.append(None)
        Connection(hidden[-1], output, Dense(Normal(mean=p["HIDDEN_OUT_MEAN"], sd=p["HIDDEN_OUT_SD"])),
                Exponential(5.0))
    return input, network, ff, rec, hidden, output
