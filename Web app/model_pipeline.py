import tensorflow as tf
import numpy as np
import os
import cv2

MODEL_PATH = r"D:\Heart attack\Web app\best_model.h5"
IMG_HEIGHT = 200
IMG_WIDTH = 490
CLASS_NAMES = [ "normal", "mi", "abnormal"]
LAST_CONV_LAYER = "conv2d_3"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_for_model(img_path):
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img / 255.0
    return np.expand_dims(img, axis=0)

def compute_gradcam(img_array):

    grad_model = tf.keras.models.Model(
        inputs=[model.input],
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))

    return heatmap

def analyze_ecg(lead_paths):

    predictions = []
    labels = []
    gradcams = []

    for path in lead_paths:
        img = preprocess_for_model(path)
        pred = model.predict(img, verbose=0)[0]

        predictions.append(pred)
        labels.append(CLASS_NAMES[np.argmax(pred)])

        heatmap = compute_gradcam(img)
        gradcams.append(heatmap)

    predictions = np.array(predictions)

    # Final diagnosis
    avg_prob = np.mean(predictions, axis=0)
    final_class = CLASS_NAMES[np.argmax(avg_prob)]

    # Severity
    mi_index = CLASS_NAMES.index("mi")
    mi_leads = np.sum(np.argmax(predictions, axis=1) == mi_index)

    total = len(lead_paths)
    severity_index = (mi_leads / total) * 100

    if severity_index <= 25:
        severity = "Mild"
    elif severity_index <= 50:
        severity = "Moderate"
    else:
        severity = "Severe"

    return {
        "diagnosis": final_class,
        "severity": severity,
        "mi_leads": int(mi_leads),
        "total_leads": total,
        "probabilities": avg_prob,
        "gradcams": gradcams 
    }