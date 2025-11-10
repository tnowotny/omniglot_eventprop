import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa  # required for affine transforms

def get_kernel(name: str):
    kernels = {
        "gaussian3": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        "sharpen": [[0,-1,0],[-1,5,-1],[0,-1,0]],
        "sobel_x": [[-1,0,1],[-2,0,2],[-1,0,1]],
        "sobel_y": [[-1,-2,-1],[0,0,0],[1,2,1]],
        "laplace": [[0,1,0],[1,-4,1],[0,1,0]],
        "none": [[0,0,0],[0,1,0],[0,0,0]],
    }
    k = np.array(kernels[name], np.float32)
    if name == "gaussian3":
        k = k / k.sum()
    return k

# -----------------------------Basic data handling ------------------------------
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

def _prep_example(alphabet, alphabet_char_id, img, lbl, target_h= 28, target_w= 28,
                  invert=True, do_minmax=True):
    img = tf.image.convert_image_dtype(img, tf.float32)
    if img.shape.rank == 3 and img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [target_h, target_w])
    img = tf.squeeze(img, axis=-1)
    if do_minmax:
        mn, mx = tf.reduce_min(img), tf.reduce_max(img)
        img = tf.where(mx > mn, (img - mn) / (mx - mn), img)
    if invert:
        img = 1.0 - img
    img = tf.clip_by_value(img, 0.0, 1.0)
    return alphabet, alphabet_char_id, img, lbl

"""
Function for loading the entire omniglot dataset (alphabet= None) or a single alphabet (e.g. alphabet="1") or a list of alphabets (e.g. alphabet= ["0", "1"]).
"""

def load_omniglot(alphabet=None, target_h= 28, target_w= 28, invert= True):
    data = []
    # TensorFlow stupidly splits the alphabets into "train" and "test" with unknown
    # reasoning for the arbitrary split; here, we allow to load any combination of alphabets
    for split in ["train", "test"]:
        ds = tfds.load("omniglot", split=split)
        ds = ds.map(lambda ex: _prep_example(ex["alphabet"], ex["alphabet_char_id"], ex["image"], ex["label"], target_h, target_w, invert))
        data.append(ds)
    if (isinstance(alphabet,str)):
        alphabet= [ alphabet ]
    images, labels = [], []
    a = set()
    i=0
    for ds in data:
        for ex in ds.as_numpy_iterator():
            i=i+1
            if alphabet is None or ex[0] in alphabet:
                a.add(ex[0])
                images.append(ex[2])
                labels.append(ex[3])
    images, labels = np.asarray(images), np.asarray(labels)
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.vectorize(remap.get)(labels)
    print(f"[Omniglot] kept {len(images)} samples, {len(uniq)} classes (alphabets loaded={a})")
    return images, labels, len(uniq)

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
def augment_images(X, mode="none", max_shift=4, max_rot=10, max_shear=0.2, zoom_range=(0.9, 1.1),contrast_range=(0.85, 1.15)):
    """Apply augmentation according to the chosen mode."""
    X = tf.convert_to_tensor(X.reshape((-1, 28, 28, 1)), dtype=tf.float32)
    out = []
    for img in X:
        if mode == "none":
            out.append(img)
            continue
        shift_x = np.random.uniform(-max_shift, max_shift)
        shift_y = np.random.uniform(-max_shift, max_shift)
        angle = np.random.uniform(-max_rot, max_rot)
        shear_x = np.random.uniform(-max_shear, max_shear)
        shear_y = np.random.uniform(-max_shear, max_shear)
        cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        zoom = np.random.uniform(*zoom_range)
        c_factor = np.random.uniform(*contrast_range)
        
        # Build affine transform according to mode
        if mode == "shift" or "shift_contrast":
            transform = [1, 0, shift_x, 0, 1, shift_y, 0, 0]
        elif mode == "shift_zoom" or "shift_zoom_contrast":
            transform = [zoom, 0, shift_x, 0, zoom, shift_y, 0, 0]    
        elif mode == "rotation":
            transform = [cos_a, -sin_a, 0, sin_a, cos_a, 0, 0, 0]
        elif mode == "rotation_shift":
            transform = [cos_a, -sin_a, shift_x, sin_a, cos_a, shift_y, 0, 0]
        elif mode == "rotation_shift_shear":
            transform = [
                cos_a + shear_x, -sin_a, shift_x,
                sin_a, cos_a + shear_y, shift_y,
                0, 0
            ]
        elif mode == "rotation_shift_shear_zoom" or "rotation_shift_shear_zoom_contrast":
            transform = [
                zoom * (cos_a + shear_x), -sin_a, shift_x,
                sin_a, zoom * (cos_a + shear_y), shift_y,
                0, 0
            ]
        
        img_aug = tfa.image.transform(img, transform, fill_mode='nearest')

        # --- Apply random contrast jitter ---

        if mode ==  "rotation_shift_shear_zoom_contrast" or "shift_zoom_contrast" or "shift_contrast":
           img_aug = tf.image.adjust_contrast(img_aug, contrast_factor=c_factor)

        # --- Clip pixel values to [0, 1] ---

        img_aug = tf.clip_by_value(img_aug, 0.0, 1.0)
        out.append(img_aug)

    X_aug = np.array([tf.squeeze(i).numpy() for i in out])
    return X_aug.reshape((-1, 28 * 28))

# --------------------------- Regularization helper ----------------------------
def apply_dropout(X, rate=0.2):
    """
    Apply dropout-like regularization to input activations or flattened images.
    Randomly sets a fraction of elements to zero (simulates neuron deactivation).

    Parameters
    ----------
    X : np.ndarray
        Input array (e.g., flattened images) of shape [N, 784].
    rate : float
        Fraction of units to drop (e.g., 0.2 = 20%).

    Returns
    -------
    np.ndarray
        Array with random elements set to zero.
    """
    if rate <= 0.0:
        return X
    mask = np.random.binomial(1, 1.0 - rate, X.shape)
    return X * mask

# --------------------------- Convolution helper ----------------------------

def convolve_images(X_flat, kernel_name="none"):
    if kernel_name == "none":
        return X_flat
    k = _get_kernel(kernel_name)
    X4 = tf.convert_to_tensor(X_flat.reshape(-1,28,28,1), dtype=tf.float32)
    k4 = tf.reshape(tf.convert_to_tensor(k), (3,3,1,1))
    Y = tf.nn.conv2d(X4, k4, strides=1, padding="SAME")
    y_min, y_max = tf.reduce_min(Y), tf.reduce_max(Y)
    Y = (Y - y_min) / (y_max - y_min + 1e-6)
    return Y.numpy().reshape(-1, 28*28)

# --------------------------- Helper ----------------------------
def get_accuracy(m):
    if isinstance(m, dict):
        for v in m.values():
            if hasattr(v, "result"): return float(v.result)
    if hasattr(m, "result"): return float(m.result)
    if isinstance(m, (float,int)): return float(m)
    return 0.0


def plot_examples(X, sy, sx, ny, nx):
    fig, ax = plt.subplots(nx,ny)
    for i in range(ny):
        for j in range(nx):
            ax[i,j].imshow(X_aug[i*nx+j,:].reshape((sy,sx)))
    plt.show()


# --------------------------- Save example images ----------------------------
def save_examples(outdir, alphabet):
    base_examples_dir = os.path.join(outdir, f"examples_alphabet_{alphabet}")
    os.makedirs(base_examples_dir, exist_ok=True)
    fig, ax = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(100):
        py, px = divmod(i, 10)
        ax[py, px].imshow(X_train[i].reshape(28, 28), cmap='gray')
        ax[py, px].axis("off")
    plt.tight_layout()
    grid_path = os.path.join(base_examples_dir, "grid_examples.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"[examples] Saved grid of examples to: {grid_path}")


# --------------------------- Plot results with smoothing and best point ------------
def moving_average(data, window_size=10):
    """
    Compute simple moving average for smoothing.
    Used to make the accuracy curve less noisy.
    """
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode="valid")

