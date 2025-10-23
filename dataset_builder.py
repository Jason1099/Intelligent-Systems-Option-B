import os, re, random 
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ------------------ config ------------------
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
IMG_H = IMG_W = 28          # match MNIST
BATCH_SIZE = 256
CUSTOM_DIR = "symbols"       # flat folder with <label>-<id>.png

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass

# ------------------ vocabulary ------------------
# Digits + operators (+, -, /, x[multiply]) + variables (w, y, z). NO variable 'x'.
VOCAB = [str(d) for d in range(10)] + ['+','-','/','x','w','y','z']
LABEL_LOOKUP = tf.keras.layers.StringLookup(vocabulary=VOCAB, num_oov_indices=0, mask_token=None)

# ------------------ helpers ------------------
LABEL_RE = re.compile(r"^([^\/\\]+?)-\d+\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)

def _norm_label(raw: str) -> str:
    raw = raw.strip()
    # Map all multiply variants to lowercase 'x' (operator)
    alias = {
        '*': 'x', 'star': 'x', 'dot': 'x', 'X': 'x',
        'plus': '+', 'minus': '-', 'slash': '/',
    }
    return alias.get(raw, raw)

def _load_mnist():
    def _map(ex):
        img = tf.cast(ex["image"], tf.float32) / 255.0
        img = tf.image.resize(img, (IMG_H, IMG_W))
        label_str = tf.strings.as_string(ex["label"])  # "0".."9"
        return img, label_str
    
    mnist_tr = tfds.load("mnist", split="train", as_supervised=False).map(_map, num_parallel_calls=AUTOTUNE)
    mnist_te = tfds.load("mnist", split="test",  as_supervised=False).map(_map, num_parallel_calls=AUTOTUNE)
    return mnist_tr, mnist_te

def _scan_custom(dir_path=CUSTOM_DIR):
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Custom folder not found: {dir_path}")
    pairs = []
    for fname in os.listdir(dir_path):
        low = fname.lower()
        if not low.endswith((".png",".jpg",".jpeg",".bmp")): continue
        m = LABEL_RE.match(fname)
        base = os.path.splitext(fname)[0]
        raw = (m.group(1) if m else base.split("-")[0]).strip()
        canon = _norm_label(raw)
        if canon not in VOCAB:
            # Skip anything unexpected
            print(f"[SKIP] {fname} -> '{canon}' not in VOCAB")
            continue
        pairs.append((os.path.join(dir_path, fname), canon))
    if not pairs:
        raise RuntimeError(f"No labeled images found in {dir_path}. Expect '<label>-<id>.png'.")
    return pairs

def _split(items, ratios=(0.8, 0.1, 0.1)):
    n = len(items); idxs = list(range(n)); random.shuffle(idxs)
    n_tr = int(ratios[0]*n); n_va = int(ratios[1]*n)
    tr = [items[i] for i in idxs[:n_tr]]
    va = [items[i] for i in idxs[n_tr:n_tr+n_va]]
    te = [items[i] for i in idxs[n_tr+n_va:]]
    return tr, va, te

def _ds_from_pairs(pairs):
    paths = tf.constant([p for p,_ in pairs])
    labels = tf.constant([s for _,s in pairs])
    def _load(path, label_str):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, (IMG_H, IMG_W), method="bilinear")
        return img, label_str

    return tf.data.Dataset.from_tensor_slices((paths, labels)).map(_load, num_parallel_calls=AUTOTUNE)

# ------------------ main method ------------------
def build_single_symbol_datasets(batch_size=BATCH_SIZE):
    """
    Returns:
      train_ds, val_ds, test_ds, vocab_list
    Each dataset yields (image[28,28,1], class_id) where class_id indexes VOCAB.
    """
    mnist_tr, mnist_te = _load_mnist()

    pairs = _scan_custom(CUSTOM_DIR)                 
    cust_tr, cust_va, cust_te = _split(pairs, (0.8, 0.1, 0.1))
    ds_ct_tr = _ds_from_pairs(cust_tr)
    ds_ct_va = _ds_from_pairs(cust_va)
    ds_ct_te = _ds_from_pairs(cust_te)

    train_union = mnist_tr.concatenate(ds_ct_tr)
    val_union   = ds_ct_va
    test_union  = mnist_te.concatenate(ds_ct_te)

    def encode(img, label_str):
        class_id = LABEL_LOOKUP(label_str)  
        return img, class_id

    train_ds = (train_union
                .map(encode, num_parallel_calls=AUTOTUNE)
                .shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
                .batch(batch_size).prefetch(AUTOTUNE))

    val_ds = (val_union
              .map(encode, num_parallel_calls=AUTOTUNE)
              .batch(batch_size).prefetch(AUTOTUNE))

    test_ds = (test_union
               .map(encode, num_parallel_calls=AUTOTUNE)
               .batch(batch_size).prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds, VOCAB

# ------------------ testing ------------------
if __name__ == "__main__":
    print("Building MNIST + symbols datasets… (operator 'x' used for multiplication; variables: w,y,z)")
    tr, va, te, vocab = build_single_symbol_datasets()
    for imgs, ids in tr.take(1):
        print("Train batch images:", imgs.shape)
        print("Train batch class_ids:", ids.shape)
        print("First 12 ids:", ids.numpy()[:12])
        break
    print("Vocabulary (index order):", vocab)
    print("Done.")