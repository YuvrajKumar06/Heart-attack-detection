from preprocessing import preprocess_ecg
from image_segmentation import segment_leads
from model_pipeline import analyze_ecg
import os
import matplotlib.pyplot as plt

def run_pipeline(image_path):

    print("\nStep 1: Preprocessing...")
    processed = preprocess_ecg(image_path)

    print("Step 2: Segmenting leads...")
    leads = segment_leads(processed)

    print("Step 3: Running model...")
    result = analyze_ecg(leads)

    print("\n========================")
    print("FINAL ECG REPORT")
    print("========================")

    print(f"Diagnosis     : {result['diagnosis'].upper()}")
    print(f"Severity      : {result['severity']}")
    print(f"MI Leads      : {result['mi_leads']}/{result['total_leads']}")

    print("\nClass Probabilities:")
    
    
    class_names = ["normal", "mi", "abnormal"]

    for i, cls in enumerate(class_names):
        print(f"{cls.upper()} → {result['probabilities'][i]*100:.2f}%")

    if "gradcams" in result:

        print("\n🔹 Generating Grad-CAM...")

        fig, axes = plt.subplots(3, 4, figsize=(12, 8))

        for i, heatmap in enumerate(result["gradcams"]):
            r = i // 4
            c = i % 4

            axes[r, c].imshow(heatmap, cmap="jet", aspect="auto")
            axes[r, c].set_title(f"Lead {i+1}")
            axes[r, c].axis("off")

        plt.tight_layout()
        plt.show()

    else:
        print("Grad-CAM not available")

    return result


if __name__ == "__main__":

    path = input("Enter ECG image path: ").strip()

    if not os.path.exists(path):
        print("File not found")
    else:
        run_pipeline(path)
