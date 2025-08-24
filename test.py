from preprocess import image_preprocessor
from segmentation import *
import os, cv2

out_dir = "digits"
path = "handWrittenDigitsTest.png"
os.makedirs(out_dir, exist_ok=True)

processor = image_preprocessor(path, size = (28, 28))
pre = processor.preprocess()
processor.show_image(pre)

seg = image_segmentation()
bboxes, crops = seg.segmentation(pre)
print(f"Found {len(bboxes)} components")

for i, crop in enumerate(crops):
    cv2.imwrite(os.path.join(out_dir, f"digit_{i:03}.png"), crop)

vis = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
for (x,y,w,h,_) in bboxes:
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imwrite(os.path.join(out_dir, "debug_boxes.png"), vis)

print(f"Saved crops and overlay to: {out_dir}")