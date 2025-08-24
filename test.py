from preprocess import image_preprocessor
from segmentation import *
import os, cv2

out_dir = "digits"
path = "handWrittenDigitsTest.png"
os.makedirs(out_dir, exist_ok=True)

# Step 1: Preprocess
processor = image_preprocessor(path, size=(28, 28))
pre = processor.preprocess()
processor.show_image(pre)

# Step 2: Segment
seg = image_segmentation(
    min_area_ratio=0.1,
    max_ar=3.0,
    margin=3,
    center=True,
    crop_size=28
)
bboxes, crops = seg.segmentation(pre)

# Step 3: Save crops
for i, crop in enumerate(crops):
    cv2.imwrite(os.path.join(out_dir, f"digit_{i:03}.png"), crop)

# Step 4: Debug overlay
vis = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
for (x,y,w,h,_) in bboxes:
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imwrite(os.path.join(out_dir, "debug_boxes.png"), vis)

print(f"Saved {len(crops)} crops and overlay to: {out_dir}")
