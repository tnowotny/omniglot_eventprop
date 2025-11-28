import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa  # required for affine transforms
import os
import matplotlib.pyplot as plt

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

def load_omniglot(alphabet_ids=None, target_h=28, target_w=28, invert=True):
    images = []
    labels = []
    alph_ids = []
    char_ids = []

    # Load TFDS metadata
    builder = tfds.builder("omniglot")
    builder.download_and_prepare()
    info = builder.info
    metadata = info.features["alphabet"].names

    # Normalize alphabet list
    if alphabet_ids is not None:
        alphabet_ids = [int(a) for a in alphabet_ids]
    else:
        alphabet_ids = list(range(len(metadata)))

    # Load full dataset (train+test merged)
    ds = tfds.load("omniglot", split="train+test", shuffle_files=False)

    # Preprocess each example
    ds = ds.map(lambda ex: _prep_example(
        ex["alphabet"],
        ex["alphabet_char_id"],
        ex["image"],
        ex["label"],
        target_h, target_w, invert
    ))

    # Convert TFDS tensors to numpy
    for alph_id, char_id, img, lbl in ds.as_numpy_iterator():

        alph_id = int(alph_id)

        # Skip alphabets not selected
        if alph_id not in alphabet_ids:
            continue

        images.append(img)
        labels.append(lbl)
        alph_ids.append(alph_id)
        char_ids.append(int(char_id))

    # Final conversion
    images = np.asarray(images).reshape(-1, 28 * 28)
    labels = np.asarray(labels)
    alph_ids = np.asarray(alph_ids)
    char_ids = np.asarray(char_ids)

    # Remap labels
    uniq = np.unique(labels)
    remap = {u: i for i, u in enumerate(uniq)}
    labels = np.vectorize(remap.get)(labels)

    print(f"[Omniglot] Loaded {len(images)} samples from alphabets {alphabet_ids}")
    print(f"           Classes={len(uniq)}")

    return images, labels, alph_ids, char_ids, len(uniq)


def show_alphabet_grids(X, alph_ids, char_ids, select_alphabets=None, save_dir=None):
    """
    For each alphabet:
      - take the first sample for each character
      - show a grid (square or rectangular)
      - optionally save to PNG
    
    Parameters
    ----------
    X : np.ndarray
        Image dataset [N, 784]
    alph_ids : np.ndarray
        Alphabet ID for each sample
    char_ids : np.ndarray
        Character ID within alphabet
    select_alphabets : list[int] or None
        Which alphabets to show. If None → all.
    save_dir : str or None
        If given → save each grid as grid_alphabet_<id>.png
    """
    if select_alphabets is None:
        select_alphabets = np.unique(alph_ids)

    X = X.reshape(-1, 28, 28)

    for a in select_alphabets:
        chars = np.unique(char_ids[alph_ids == a])
        num_chars = len(chars)

        print(f"\n[Grid] Alphabet {a} has {num_chars} characters")

        # Collect one representative image per character
        samples = []
        for c in chars:
            idx = np.where((alph_ids == a) & (char_ids == c))[0]
            samples.append(X[idx[0]])   # take the first sample

        # Determine grid layout
        cols = int(np.ceil(np.sqrt(num_chars)))
        rows = int(np.ceil(num_chars / cols))

        # Plot
        fig, ax = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        ax = np.array(ax).reshape(rows, cols)

        k = 0
        for r in range(rows):
            for c in range(cols):
                ax[r, c].axis("off")
                if k < num_chars:
                    ax[r, c].imshow(samples[k], cmap="gray")
                k += 1

        plt.suptitle(f"Alphabet {a}: {num_chars} characters", fontsize=14)

        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"grid_alphabet_{a}.png")
            plt.savefig(path, dpi=150)
            print(f"Saved grid for alphabet {a}: {path}")
            plt.close()
        else:
            plt.show()

def stratified_split(X, y, val_fraction=0.1):
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], train_idx, val_idx

# --------------------------- Augmentation functions ----------------------------
def augment_images(X, mode="none", max_shift=4, max_rot=10, max_shear=0.25, zoom_range=(0.9, 1.1),contrast_range=(0.85, 1.15)):
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
        if mode in ["shift", "shift_contrast"]:
            transform = [1, 0, shift_x, 0, 1, shift_y, 0, 0]
        elif mode in ["shift_zoom", "shift_zoom_contrast"]:
            transform = [zoom, 0, shift_x, 0, zoom, shift_y, 0, 0]
        
        elif mode in ["zoom", "zoom_contrast"]:
            transform = [zoom, 0, 0, 0, zoom, 0, 0, 0]    
        
        elif mode in ["rotation", "rotation_contrast"]:
            transform = [cos_a, -sin_a, 0, sin_a, cos_a, 0, 0, 0]
        
        elif mode in ["rotation_zoom", "rotation_zoom_contrast"]:
            transform = [zoom * cos_a, -sin_a, 0, sin_a, zoom * cos_a, 0, 0, 0]    
        
        elif mode in ["rotation_shift", "rotation_shift_contrast"]:
            transform = [cos_a, -sin_a, shift_x, sin_a, cos_a, shift_y, 0, 0]
        
        elif mode in ["rotation_shift_zoom", "rotation_shift_zoom_contrast"]:
            transform = [zoom * cos_a, -sin_a, shift_x, sin_a, zoom * cos_a, shift_y, 0, 0]    
        
        elif mode in ["shear", "shear_contrast"]:
            cx, cy =14, 14
            transform = [
                1.0, shear_x, -shear_x * cy, 
                shear_y, 1.0, -shear_y * cx,
                0.0, 0.0
            ]
        
        elif mode in ["shear_zoom", "shear_zoom_contrast"]:
            cx, cy =14, 14
            
            tx = -zoom * cx -shear_x * cy + cx
            ty = -shear_y * cx - zoom * cy + cy
            
            # Shear + Zoom matrix around center
            transform = [
                zoom, shear_x, tx, 
                shear_y, zoom, ty,
                0.0, 0.0
            ]
        
        elif mode in ["shear_shift", "shear_shift_contrast"]:
            cx, cy = 14, 14
            transform = [
                1.0, shear_x, -shear_x * cy + shift_x,
                shear_y, 1.0, -shear_y * cx + shift_y,
                0.0, 0.0
            ]
        
        elif mode in ["shear_shift_zoom", "shear_shift_zoom_contrast"]:
            cx, cy = 14, 14
            
            tx = -zoom * cx -shear_x * cy + cx
            ty = -shear_y * cx - zoom * cy + cy
            
            transform = [
                zoom, shear_x, tx + shift_x,
                shear_y, zoom, ty + shift_y,
                0.0, 0.0
            ]
        
        elif mode in ["rotation_shear", "rotation_shear_contrast"]:
            cx, cy = 14, 14
            
            # rotate then shear
            transform = [
                cos_a + shear_x, -sin_a, -shear_x * cy,
                sin_a, cos_a + shear_y, -shear_y * cx,
                0.0, 0.0
            ]
        
        elif mode in ["rotation_shear_zoom", "rotation_shear_zoom_contrast"]:
            
            cx, cy = 14, 14
            
            # Combined matrix A for rotation + shear + zoom            
            A00 = zoom * (cos_a + shear_x)    
            A01 = zoom * (-sin_a)    
            A10 = zoom * (sin_a)    
            A11 = zoom * (cos_a + shear_y)
            
            # Correct center compensation    
            tx = -A00 * cx - A01 * cy + cx    
            ty = -A10 * cx - A11 * cy + cy
            transform = [
                A00, A01, tx,        
                A10, A11, ty,        
                0.0, 0.0
            ]
            
            
        elif mode in ["rotation_shift_shear", "rotation_shift_shear_contrast"]:
            
            cx, cy = 14, 14
          
            transform = [
                cos_a + shear_x, -sin_a, -shear_x * cy + shift_x,
                sin_a, cos_a + shear_y, -shear_y * cx + shift_y,
                0.0, 0.0
            ]
                          
        elif mode in ["rotation_shift_shear_zoom", "rotation_shift_shear_zoom_contrast"]:
            
            cx, cy = 14, 14
            
            # Build combined transform coefficients
            A00 = zoom * (cos_a + shear_x)
            A01 = zoom * (-sin_a)    
            A10 = zoom * (sin_a)    
            A11 = zoom * (cos_a + shear_y)

            # Correct center compensation + shift
            tx = -A00 * cx - A01 * cy + cx + shift_x
            ty = -A10 * cx - A11 * cy + cy + shift_y

            transform = [
                A00, A01, tx,        
                A10, A11, ty,        
                0.0, 0.0
            ]
        
        img_aug = tfa.image.transform(img, transform, fill_mode='nearest')

        # --- Apply random contrast jitter ---

        if "contrast" in mode: 
               
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
def save_examples(X, alph_ids, char_ids, outdir, select_alphabets=None, max_per_char=200):

    """
    Save all samples separated by alphabet and character ID.

    Structure:
      outdir/
         alphabet_0/
             char_0/
                 sample_000.png
                 sample_001.png
                 ...
                 grid.png
             char_1/
                 ...
    
    Parameters
    ----------
    X : np.ndarray
        Image data, shape [N, 784]
    alph_ids : np.ndarray
        Alphabet numerical ID for each sample
    char_ids : np.ndarray
        Character ID within the alphabet for each sample
    outdir : str
        Base folder
    select_alphabets : list[int]
        Which alphabet IDs to save (if None → save all)
    max_per_char : int
        How many samples to save per character (default 200)
    """

    os.makedirs(outdir, exist_ok=True)

    X_reshaped = X.reshape(-1, 28, 28)

    if select_alphabets is None:
        select_alphabets = np.unique(alph_ids)

    for a in select_alphabets:
        print(f"[save_examples] Processing alphabet {a}")

        a_dir = os.path.join(outdir, f"alphabet_{a}")
        os.makedirs(a_dir, exist_ok=True)

        # find all characters inside this alphabet
        chars_in_a = np.unique(char_ids[alph_ids == a])

        for c in chars_in_a:
            c_dir = os.path.join(a_dir, f"char_{c}")
            os.makedirs(c_dir, exist_ok=True)

            # indices for this specific character
            idx = np.where((alph_ids == a) & (char_ids == c))[0]

            if len(idx) == 0:
                continue

            num_to_save = min(max_per_char, len(idx))

            for i, sample_i in enumerate(idx[:num_to_save]):
                img = X_reshaped[sample_i]
                path = os.path.join(c_dir, f"sample_{i:03d}.png")
                plt.imsave(path, img, cmap="gray")

            # save 10×10 grid of first 100
            grid_count = min(100, len(idx))
            fig, ax = plt.subplots(10, 10, figsize=(7, 7))

            for k in range(grid_count):
                py, px = divmod(k, 10)
                ax[py, px].imshow(X_reshaped[idx[k]], cmap="gray")
                ax[py, px].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(c_dir, "grid.png"), dpi=150)
            plt.close()

            print(f"  Saved {num_to_save} samples for char {c} → {c_dir}")

    print(f"[save_examples] Done. Output saved under {outdir}")

'''
def save_examples(X, y, alphabets, outdir, max_per_alphabet=100):

    os.makedirs(outdir, exist_ok=True)

    for a in alphabets:
        alphabet_dir = os.path.join(outdir, f"alphabet_{a}")
        os.makedirs(alphabet_dir, exist_ok=True)
        
        indices = np.where(y == int(a))[0]
        if len(indices) == 0:
            print(f"[save_examples] No samples found for alphabet {a}")
            continue
        num_to_save = min(max_per_alphabet, len(indices))
        chosen = np.random.choice(indices, num_to_save, replace=False)

        # ---- Save individual samples ----
        for i, idx in enumerate(chosen):
            img = X[idx].reshape(28, 28)
            path = os.path.join(alphabet_dir, f"sample_{i:03d}.png")
            plt.imsave(path, img, cmap="gray")

        # ---- Save grid of 100 examples (10x10) ----
        fig, ax = plt.subplots(10, 10, figsize=(8, 8))
        for j in range(min(100, len(chosen))):
            py, px = divmod(j, 10)
            ax[py, px].imshow(X[chosen[j]].reshape(28, 28), cmap='gray')
            ax[py, px].axis("off")

        plt.tight_layout()
        grid_path = os.path.join(alphabet_dir, f"alphabet_{a}_grid_examples.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()

        print(f"[save_examples] Saved {num_to_save} samples and grid to: {alphabet_dir}")

'''

def save_augmented_grid(X_aug, alph_ids, char_ids, outdir, epoch,
                        alphabet_id, augment_name="none", max_chars=200):
    """
    Save a single grid (combined image) of augmented samples
    from one selected alphabet. Used to visually inspect augmentation.

    Folder structure:
        outdir/
            augmented_grid_alphabet_0.png

    Parameters
    ----------
    X_aug : np.ndarray
        Augmented images, shape [N, 784]
    alph_ids : np.ndarray
        Alphabet ID for each original sample (must correspond to X_aug)
    char_ids : np.ndarray
        Character ID inside the alphabet
    outdir : str
        Folder where the grid will be saved
    epoch : int
        Current epoch number (we save only when epoch == 1)
    alphabet_id : int
        The alphabet to visualize
    max_chars : int
        Maximum number of augmented samples to include in the grid
    """

    # Create output directory if needed
    os.makedirs(outdir, exist_ok=True)

    # Reshape from flat to 28×28
    X_r = X_aug.reshape(-1, 28, 28)

    # Select only samples belonging to the requested alphabet
    idx = np.where(alph_ids == alphabet_id)[0]
    if len(idx) == 0:
        print(f"[aug grid] No samples for alphabet {alphabet_id}")
        return

    # Limit how many images to show
    idx = idx[:max_chars]
    N = len(idx)

    # Compute grid size
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))

    # Create figure
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    ax = np.array(ax).reshape(rows, cols)

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax[r, c].axis("off")
            if k < N:
                ax[r, c].imshow(X_r[idx[k]], cmap="gray")
            k += 1

    plt.suptitle( f"Augmented Alphabet {alphabet_id} | Augmentation: {augment_name} | Epoch {epoch}",
        fontsize=14
        )

    # Save file
    save_path = os.path.join(outdir, f"augmented_grid_alphabet_{alphabet_id}_{augment_name}_.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"[aug grid] Saved augmented grid for alphabet {alphabet_id} {augment_name} : {save_path}")

# --------------------------- Plot results with smoothing and best point ------------
def moving_average(data, window_size=10):
    """
    Compute simple moving average for smoothing.
    Used to make the accuracy curve less noisy.
    """
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode="valid")

