import os
import cv2
import pandas as pd

DATA_ROOT = "./data"
csv_path = os.path.join(DATA_ROOT, "val.csv")  # or "train.csv"
df = pd.read_csv(csv_path)

# pick a few rows to inspect
for idx in [0, 10, 50]:
    row = df.iloc[idx]
    image_name = row["Image name"]          # e.g. image_00001_img.jpg
    cls_name = row["Class"]                 # e.g. VenusExpress

    # image path: data/images/Class/val/image_00001_img.jpg
    split = "val"                           # or "train" depending on csv_path
    img_path = os.path.join(
        DATA_ROOT, "images", cls_name, split, image_name
    )

    # mask path: data/mask/Class/val/image_00001_layer.jpg
    mask_name = image_name.replace("_img.jpg", "_layer.jpg")
    mask_path = os.path.join(
        DATA_ROOT, "mask", cls_name, split, mask_name
    )

    print(f"Sample {idx}:")
    print("  img :", img_path)
    print("  mask:", mask_path)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if img is None:
        print("  [ERROR] could not read image")
        continue
    if mask is None:
        print("  [ERROR] could not read mask")
        continue

    # overlay
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    alpha = 0.5
    overlay = img.copy()
    non_black = (mask_resized.sum(axis=2) > 0)
    overlay[non_black] = cv2.addWeighted(
        img[non_black], 1 - alpha, mask_resized[non_black], alpha, 0
    )

    os.makedirs("debug_overlays", exist_ok=True)
    out_path = os.path.join("debug_overlays", f"overlay_{idx}.png")
    cv2.imwrite(out_path, overlay)
    print(f"  [OK] saved overlay to {out_path}")
