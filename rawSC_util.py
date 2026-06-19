import numpy as np
import copy
import matplotlib.pyplot as plt
from ml_genn.callbacks import VarRecorder

class Shift:
    def __init__(self, f_shift, num_input):
        self.f_shift = f_shift
        self.num_input = num_input

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        # Shift events
        in_copy = copy.deepcopy(inp)
        for i in range(inp.shape[0]):
            in_copy[i,:,:] = 0.0
            s = np.random.randint(-self.f_shift, self.f_shift)
            if s < 0:
                in_copy[i,:,:s] = inp[i,:,-s:]
            elif s > 0:
                in_copy[i,:,s:] = inp[i,:,:-s]
        return in_copy


def extract_embeddings(compiled_net, input, output, X_test, layer, batch_size, var):
    """run inference, return membrane voltage embeddings."""
    print(f"Extracting embeddings")
    with compiled_net:
        n_batches = int(np.ceil(len(X_test) / batch_size))
        all_embeddings = []
        pops = compiled_net.genn_model.neuron_populations
        the_pop = pops[layer]
        for batch_idx, start in enumerate(range(0, len(X_test), batch_size)):
            batch_img = X_test[start : start + batch_size]
            compiled_net.evaluate({input: batch_img},{output: np.zeros(batch_img.shape[0])},callbacks=[])
            the_pop.vars[var].pull_from_device()
            all_embeddings.append(the_pop.vars[var].values)
        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Done: shape {embeddings.shape}, "
              f"norm mean={norms.mean():.3f} std={norms.std():.3f} "
              f"min={norms.min():.3f} max={norms.max():.3f}")
    return embeddings

def extract_detailed_embeddings(compiled_net, input, output, X_test, layer, batch_size,var):
    """run inference with VarRecorder, return membrane voltage traces embeddings."""
    print(f"Extracting detailed embeddings")
    with compiled_net:
        n_batches = int(np.ceil(len(X_test) / batch_size))
        all_embeddings = []
        callbacks = [ VarRecorder(layer, var, "v_emb")]
        for batch_idx, start in enumerate(range(0, len(X_test), batch_size)):
            batch_img = X_test[start : start + batch_size]
            n = len(batch_img)
            metrics, cb_data = compiled_net.evaluate({input: batch_img},{output: np.zeros(batch_img.shape[0])},callbacks=callbacks)
            all_embeddings.extend(cb_data["v_emb"])
        embeddings = np.asarray(all_embeddings)
        embeddings = embeddings.reshape((-1,embeddings.shape[1]*embeddings.shape[2]))
        print(embeddings.shape)
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Done: shape {embeddings.shape}, "
              f"norm mean={norms.mean():.3f} std={norms.std():.3f} "
              f"min={norms.min():.3f} max={norms.max():.3f}")
    return embeddings

def l2_normalise(embeddings):
    """L2-normalise each row to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms

def sample_episode(embeddings, labels, speakers, n_way, k_shot):
    """Sample N_WAY classes, K_SHOT support and the rest query images each.
    each example class id from the same speaker.
    """

    N = len(labels)
    num_s = k_shot+1 # need at least k_shot plus one examples to include a set
    used = np.zeros(np.max(labels)+1)
    sup_emb, sup_lb = [], []
    qry_emb, qry_lb = [], []
    for i in range(n_way):
        idx = []
        choice = np.random.randint(0, N, 1)
        while (len(idx) < num_s):
            choice += 1
            if choice >= N:
                choice = 0
            while used[labels[choice]] > 0:
                choice += 1
                if choice >= N:
                    choice = 0
                    print(f"looped: i == {i}")
            idx = np.where(np.logical_and(labels == labels[choice], speakers == speakers[choice]))[0]
        used[labels[choice]] = 1
        sup_emb.extend(embeddings[idx[:k_shot]])
        sup_lb += [i] * k_shot        
        qry_emb.extend(embeddings[idx[k_shot:]])
        qry_lb += [i] * len(idx[k_shot:])
    return np.asarray(sup_emb), np.array(sup_lb), np.asarray(qry_emb), np.array(qry_lb)

def run_episode(embeddings, labels, speakers, n_way, k_shot):
    """One episode: Euclidean nearest-centroid classifier."""
    sup_emb, sup_lb, qry_emb, qry_lb = sample_episode(embeddings, labels, speakers, n_way, k_shot)
    prototypes = np.array([sup_emb[sup_lb == c].mean(0) for c in range(n_way)])
    dists = ((qry_emb[:, None] - prototypes[None]) ** 2).sum(-1)
    return (np.argmin(dists, axis=1) == qry_lb).mean()

def run_episode_cosine(embeddings, labels, speakers,
                       n_way, k_shot):
    """One episode: cosine nearest-centroid (per-episode L2 normalisation)."""
    sup_emb, sup_lb, qry_emb, qry_lb = sample_episode(embeddings, labels, speakers, n_way, k_shot)

    sup_emb = sup_emb / np.maximum(np.linalg.norm(sup_emb, axis=1, keepdims=True), 1e-8)
    qry_emb = qry_emb / np.maximum(np.linalg.norm(qry_emb, axis=1, keepdims=True), 1e-8)

    prototypes = np.array([sup_emb[sup_lb == c].mean(0) for c in range(n_way)])
    dists = ((qry_emb[:, None] - prototypes[None]) ** 2).sum(-1)
    return (np.argmin(dists, axis=1) == qry_lb).mean()
