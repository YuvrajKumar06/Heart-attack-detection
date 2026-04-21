from PIL import Image
import os

def segment_leads(image_path, output_folder="temp/leads"):

    os.makedirs(output_folder, exist_ok=True)

    img = Image.open(image_path)
    img = img.resize((2137, 1572))

    coords = [
        (150, 340, 640, 540), (645, 340, 1135, 540),
        (1140, 340, 1630, 540), (1650, 340, 2140, 540),

        (150, 640, 640, 840), (645, 640, 1135, 840),
        (1140, 640, 1630, 840), (1650, 640, 2140, 840),

        (150, 950, 640, 1150), (645, 950, 1135, 1150),
        (1140, 950, 1630, 1150), (1650, 950, 2140, 1150)
    ]

    lead_paths = []

    for i, c in enumerate(coords, 1):
        crop = img.crop(c)
        path = os.path.join(output_folder, f"lead_{i}.jpg")
        crop.convert("RGB").save(path)
        lead_paths.append(path)

    return lead_paths