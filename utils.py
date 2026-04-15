import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa  # required for affine transforms
import matplotlib.pyplot as plt
import scipy

# parse the command line alphabet specification
def parse_alphabet_spec(spec: str):
    """Parse alphabet argument: '0-4' → [0,1,2,3,4], '0,2,5' → [0,2,5]"""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    result = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            result.extend(range(min(a, b), max(a, b) + 1))
        else:
            result.append(int(p))
    return sorted(set(result))

# pre-processing of the omniglot images as procided by tensorflow
def _prep_example(ex):
    print(ex)
    if ex["image"].shape.rank == 3 and ex["image"].shape[-1] == 3:
        ex["image"] = tf.image.rgb_to_grayscale(ex["image"])
    ex["image"] = tf.squeeze(ex["image"], axis=-1)
    ex["image"] = 255 - ex["image"]
    return ex

# load the omniglot dataset from tensorflow
def load_omniglot(split="train"):
    images = []
    labels = []
    alph_ids = []
    char_ids = []

    # Load TFDS metadata
    builder = tfds.builder("omniglot")
    builder.download_and_prepare()
    info = builder.info
    alph_names = info.features["alphabet"].names

    # Load full dataset (split == "train" or == "test")
    ds = tfds.load("omniglot", split=split, shuffle_files=False)
    # Preprocess each example
    ds = ds.map(lambda ex: _prep_example(ex)) 
    # Convert TFDS tensors to numpy
    ds = tfds.as_numpy(ds)
    for ex in ds:
        images.append(ex["image"])
        labels.append(ex["label"])
        alph_ids.append(int(ex["alphabet"]))
        char_ids.append(int(ex["alphabet_char_id"]))

    # Final conversion
    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels)
    alph_ids = np.asarray(alph_ids)
    char_ids = np.asarray(char_ids)

    # Remap labels
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.vectorize(remap.get)(labels)

    print(f"[Omniglot] Loaded {len(images)} samples from {split}")
    print(f"           Classes={len(uniq)}")

    return images, labels, alph_ids, char_ids, remap, len(uniq)


# --------------------------- make a validation split ----------------------------
def validation_split(X, y, split= 0.1):
    N = len(y)
    ids = np.arange(N)
    np.random.shuffle(ids)
    N_val = int(N*split)
    N_train = N-N_val
    X_train = X[ids[:N_train],:,:]
    y_train = y[ids[:N_train]]
    X_val = X[ids[N_train:],:,:]
    y_val = y[ids[N_train:]]
    return X_train, X_val, y_train, y_val

# --------------------- make a stratified validation split ---------------------- 
def stratified_split(X, y, val_fraction=0.1):
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# --------------------------- Augmentation functions ----------------------------
def augment_images(X, augmentations=[], max_shift=4, max_rot=10, max_shear=0.1, zoom_range=(0.9, 1.1), DEBUG=False):
    """Apply augmentation according to the chosen mode."""
    X_aug = X.copy()
    X_new = []
    for img in X_aug:
        if DEBUG:
            plt.figure()
            plt.imshow(img)
            plt.colorbar()
        shift_x = np.random.uniform(-max_shift, max_shift)
        shift_y = np.random.uniform(-max_shift, max_shift)
        angle = np.random.uniform(-max_rot, max_rot)
        shear_x = np.random.uniform(-max_shear, max_shear)
        shear_y = np.random.uniform(-max_shear, max_shear)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        if DEBUG:
            print(f"cos_a: {cos_a}, sin_a: {sin_a}")
        zoom = np.random.uniform(*zoom_range)

        A = np.identity(2)
        offset = [0.0, 0.0]
        for aug in augmentations:
            if aug == "shift":
                offset = [shift_x, shift_y]
            if aug == "rotate":
                A = np.matmul(A, np.asarray([[cos_a, -sin_a],[sin_a, cos_a]]))
            if aug == "zoom":
                A = np.matmul(A, np.asarray([[zoom, 0.0],[0.0, zoom]]))
            if aug == "shear":
                A = np.matmul(A, np.asarray([[ 1.0, 0.0], [ shear_x, 1.0 ]]))
                A = np.matmul(A, np.asarray([[ 1.0, shear_y], [ 0.0, 1.0 ]]))
        img = scipy.ndimage.affine_transform(img, A, offset)
        img = np.clip(img, 0, 255)
        if DEBUG:
            plt.figure()
            plt.imshow(img,vmin= 0, vmax= 255)
            plt.colorbar()
            plt.show()
        X_new.append(img)
       
    return np.asarray(X_new)

def rescale_images(X, ht, wd, DEBUG=False):
    X_c = X.copy()
    X_re = []
    for img in X_c:
        X_re.append(scipy.ndimage.zoom(img, [ht/img.shape[0], wd/img.shape[0]]))
        if DEBUG:
            print(X_re[-1].shape)
            plt.figure()
            plt.imshow(X_re[-1],vmin= 0, vmax= 255)
            plt.colorbar()
            plt.show()
    return np.asarray(X_re)

def show_example(X):
    fix, ax = plt.subplots(10,10)
    for i in range(10):
        for j in range(10):
            id = i*10+j
            ax[i,j].imshow(X[id])
    plt.show()
