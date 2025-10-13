import cv2
import numpy as np

class image_segmentation:
    def __init__(self, min_area_ratio=0.15, max_ar=3, crop_size=28, margin=3, center=True):
        self.min_area_ratio = min_area_ratio
        self.max_ar = max_ar
        self.crop_size = crop_size
        self.margin = margin
        self.center = center

    def segmentation(self, gray):
        bw = self._binarize_otsu(gray)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num_labels <= 1:
            return [], []
        
        bboxes = self._filter_components(stats, num_labels)
        bboxes.sort(key=lambda b: b[0])
        crops = self._extract_crops(bw, bboxes)

        return bboxes, crops
    
    def _binarize_otsu(self, gray):
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        bw = cv2.medianBlur(bw, 3)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

        return bw
    
    def _filter_components(self, stats, num_labels):
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        if not areas:
            return []
        med_area = np.median(areas) if areas else 0
        if med_area <= 0:
            return []

        bboxes = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_area_ratio * med_area:
                continue

            ar = max(w / max(h, 1), h / max(w, 1))
            if ar > self.max_ar:
                continue

            bboxes.append((x, y, w, h, area))
        
        return bboxes

    def _extract_crops(self, bw, bboxes):
        crops = []
        for (x, y, w, h, _) in bboxes:
            x0 = max(0, x - self.margin)
            y0 = max(0, y - self.margin)
            x1 = min(bw.shape[1], x + w + self.margin)
            y1 = min(bw.shape[0], y + h + self.margin)
            digit = bw[y0:y1, x0:x1]
            
            H, W = digit.shape
            s = max(H, W)
            pad_t = (s - H) // 2
            pad_b = s - H - pad_t
            pad_l = (s - W) // 2
            pad_r = s - W - pad_l
            sq = cv2.copyMakeBorder(digit, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0)

            if self.center:
                ys, xs = np.where(sq > 0)
                if len(xs):
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    tx = sq.shape[1] // 2 - cx
                    ty = sq.shape[0] // 2 - cy
                    M = np.float32([[1, 0 , tx], [0, 1, ty]])
                    sq = cv2.warpAffine(sq, M, (sq.shape[1], sq.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
            
            crop = cv2.resize(sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            crop = (crop > 0).astype(np.uint8) * 255
            crops.append(crop)
        
        return crops