import os
from preprocess import image_preprocessor
from segmentation import image_segmentation
import cv2

if __name__ == '__main__':
    out_dir = "digits"
    path = "handWrittenDigitsTest.png"
    os.makedirs(out_dir, exist_ok=True)

    processor = image_preprocessor(path, size=(28, 28))
    pre = processor.preprocess()
    # processor.show_image(pre)

    seg = image_segmentation(min_area_ratio=0.1, max_ar=4.0, margin=5)
    bboxes, crops = seg.segmentation(pre)

    for i, crop in enumerate(crops):
        cv2.imwrite(os.path.join(out_dir, f"digit_{i:03}.png"), crop)

    vis = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h, area) in enumerate(bboxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(os.path.join(out_dir, "debug_boxes.png"), vis)
    
    _, debug_binary = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out_dir, "debug_binary.png"), debug_binary)

    print(f"Saved {len(crops)} crops and debug images to: {out_dir}")