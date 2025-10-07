import cv2
import numpy as np

class image_segmentation:
    def __init__(self, min_area_ratio=0.1, max_ar=4.0, crop_size=28, margin=5, center=True):
        self.min_area_ratio = min_area_ratio
        self.max_ar = max_ar
        self.crop_size = crop_size
        self.margin = margin
        self.center = center

    def segmentation(self, gray):
        # Expect gray image
        bw = self._best_binarization(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num_labels <= 1:
            return [], []

        bboxes = self._filter_components(stats, num_labels, gray.shape)
        bboxes.sort(key=lambda b: b[0])
        crops = self._extract_crops(bw, bboxes)
        return bboxes, crops

    def _best_binarization(self, gray):
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        _, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
        mean_val = np.mean(gray)
        _, bw3 = cv2.threshold(gray, mean_val, 255, cv2.THRESH_BINARY_INV)

        methods = [bw1, bw2, bw3]
        best_bw = bw1
        best_score = -1
        for bw in methods:
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            n = max(0, num_labels - 1)

            score = max(0, 20 - abs(n - 5)) 
            if score > best_score:
                best_score = score
                best_bw = bw
        return best_bw

    def _filter_components(self, stats, num_labels, image_shape):
        if num_labels <= 1:
            return []
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
        widths = [stats[i, cv2.CC_STAT_WIDTH] for i in range(1, num_labels)]
        if not areas:
            return []
        med_area = np.median(areas)
        med_height = np.median(heights)
        med_width = np.median(widths)

        min_area_threshold = max(50, self.min_area_ratio * med_area)
        min_height = max(10, 0.3 * med_height)
        min_width = max(5, 0.2 * med_width)
        image_area = image_shape[0] * image_shape[1]
        max_area_threshold = 0.3 * image_area

        bboxes = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < min_area_threshold or area > max_area_threshold:
                continue
            if h < min_height or w < min_width:
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
            if digit.size == 0:
                continue
            H, W = digit.shape
            s = max(H, W)
            s = int(s * 1.2)
            pad_t = (s - H) // 2
            pad_b = s - H - pad_t
            pad_l = (s - W) // 2
            pad_r = s - W - pad_l
            sq = cv2.copyMakeBorder(digit, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0)

            if self.center:
                ys, xs = np.where(sq > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    tx = sq.shape[1] // 2 - cx
                    ty = sq.shape[0] // 2 - cy
                    tx = max(-sq.shape[1] // 4, min(sq.shape[1] // 4, tx))
                    ty = max(-sq.shape[0] // 4, min(sq.shape[0] // 4, ty))
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    sq = cv2.warpAffine(sq, M, (sq.shape[1], sq.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)

            crop = cv2.resize(sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            crop = (crop > 0).astype(np.uint8) * 255
            crops.append(crop)
        return crops
