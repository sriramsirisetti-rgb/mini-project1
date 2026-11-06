"""Train a CNN on the PlantVillage dataset.

Usage:
    python train.py --data data/PlantVillage --arch mobilenet   # transfer learning (recommended)
    python train.py --data data/PlantVillage --arch custom      # plain CNN from scratch
"""
import argparse, json, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = 224
AUTOTUNE = tf.data.AUTOTUNE


def collect(data_dir):
    root = pathlib.Path(data_dir)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in (root / c).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p)); labels.append(i)
    return classes, np.array(paths), np.array(labels)


def make_ds(paths, labels, batch, training):
    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG))
        return img, y
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch)
    return ds.prefetch(AUTOTUNE)


def augment_block():
    # Active only during training; mimics real-world phone photos.
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2, value_range=(0, 255)),
    ], name="augmentation")


def build_mobilenet(n_classes):
    base = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = keras.Input((IMG, IMG, 3))              # raw 0-255 pixels
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)  # MobileNetV2 preprocessing, baked into the model
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), base


def build_custom(n_classes):
    inp = keras.Input((IMG, IMG, 3))
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out), None


def compile_model(m, lr):
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PlantVillage")
    ap.add_argument("--arch", choices=["mobilenet", "custom"], default="mobilenet")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--finetune_epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    classes, paths, labels = collect(a.data)
    print(f"{len(classes)} classes, {len(paths)} images")

    # Stratified 70 / 15 / 15 split
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=42)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    train_ds = make_ds(p_tr, y_tr, a.batch, True)
    val_ds = make_ds(p_va, y_va, a.batch, False)
    test_ds = make_ds(p_te, y_te, a.batch, False)

    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_tr)
    class_weight = {i: w for i, w in enumerate(cw)}

    model, base = (build_mobilenet if a.arch == "mobilenet" else build_custom)(len(classes))
    cbs = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")]

    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=class_weight, callbacks=cbs)
    hist = {k: list(v) for k, v in h1.history.items()}

    if base is not None and a.finetune_epochs > 0:      # fine-tune top of the backbone
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.finetune_epochs,
                       class_weight=class_weight, callbacks=cbs)
        for k, v in h2.history.items():
            hist[k] += list(v)

    # ---- Evaluation on the held-out test set ----
    pathlib.Path("reports").mkdir(exist_ok=True)
    pathlib.Path("model").mkdir(exist_ok=True)
    probs = model.predict(test_ds)
    y_pred = probs.argmax(1)
    report = classification_report(y_te, y_pred, target_names=classes, digits=3)
    print(report)
    open("reports/classification_report.txt", "w").write(report)

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(12, 12)); plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix (test set)"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=120); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(hist["accuracy"], label="train"); plt.plot(hist["val_accuracy"], label="val")
    plt.legend(); plt.title("Accuracy"); plt.xlabel("epoch")
    plt.savefig("reports/training_curve.png", dpi=120); plt.close()

    model.save("model/plant_model.keras")
    json.dump(classes, open("model/class_names.json", "w"), indent=2)
    print("Saved model/plant_model.keras and model/class_names.json")


if __name__ == "__main__":
    main()
