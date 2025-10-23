import cv2
import numpy as np
import os, json

class image_segmentation:
    def __init__(self,
                min_area_ratio=0.1,
                max_ar=4.0,
                crop_size=28,
                margin=5,
                center=True,
                # --- group & line params ---
                line_v_overlap=0.50,       # vertical overlap threshold to consider same line
                line_y_tol_ratio=0.35,     # y-distance tolerance when aligning text baselines
                run_gap_ratio=0.35,        # max horizontal gap (as ratio of avg width) for grouping
                run_y_overlap=0.45,        # vertical overlap threshold for horizontal grouping
                # --- thin/tall character handling ---
                keep_tall_thin=True,       # keep tall, skinny components like '1' or '/'
                thin_min_height_ratio=0.35,# minimum height ratio to keep tall-thin boxes
                thicken_ones=True          # dilate vertically to connect thin strokes (fixes '1')
                ):
        """
        Initialize default parameters and metadata placeholders
        """
        # ---- general segmentation parameters ----
        self.min_area_ratio = min_area_ratio
        self.max_ar = max_ar
        self.crop_size = crop_size
        self.margin = margin
        self.center = center

        # ---- grouping & line detection ----
        self.line_v_overlap = line_v_overlap
        self.line_y_tol_ratio = line_y_tol_ratio
        self.run_gap_ratio = run_gap_ratio
        self.run_y_overlap = run_y_overlap

        # ---- tall/skinny detection ----
        self.keep_tall_thin = keep_tall_thin
        self.thin_min_height_ratio = thin_min_height_ratio
        self.thicken_ones = thicken_ones

        # ---- placeholders for debugging/metadata ----
        self.last_bw = None              # last binarized image used for segmentation
        self.last_filtered_boxes = []    # store filtered bounding boxes (atoms)
        self.group_manifest = {}         # manifest storing order and grouping data


    def segmentation(self, gray, export=True, out_dir="objects_export"):
        """Main segmentation pipeline:
        1. Binarize the image
        2. Detect connected components (digits)
        3. Filter valid components
        4. Group by line + build manifest
        5. Return ordered boxes and crops
        """
        # --- Step 1: binarize image (invert to make digits white on black) ---
        bw = self._best_binarization(gray)

        # --- Step 2: dilate thin strokes vertically (helps '1') ---
        if self.thicken_ones:
            vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            bw = cv2.dilate(bw, vkernel, iterations=1)

        # --- Step 3: light horizontal dilation (helps '-' or broken digits) ---
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        bw = cv2.dilate(bw, hkernel, iterations=1)

        self.last_bw = bw  # store binary for later use (e.g., cropping)

        # --- Step 4: find connected components (potential digits) ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num_labels <= 1:
            # no foreground detected
            print("No components found")
            self.last_filtered_boxes = []
            self.group_manifest = {"image_size":[bw.shape[1], bw.shape[0]], "components":[], "groups":[]}
            return [], []

        print(f"Found {num_labels-1} connected components")

        # --- Step 5: filter out noise or invalid shapes ---
        bboxes = self._filter_components(stats, num_labels, gray.shape)
        self.last_filtered_boxes = bboxes[:]  # keep for debugging

        # --- Step 6: group boxes into lines, then left→right order within each line ---
        lines_only = self._group_lines_only(bboxes)
        ordered_bboxes = [b for line in lines_only for b in line]  # flatten lines in reading order

        # --- Step 7: build metadata manifest (lines, groups, reading order) ---
        self.group_manifest = self._build_group_manifest(lines_only, bboxes, bw.shape)

        if export:
            self._export_objects(gray, out_dir)

        # --- Step 8: crop each bounding box to save as training image ---
        crops = self._extract_crops(bw, ordered_bboxes)
        return ordered_bboxes, crops, self.group_manifest


    def _best_binarization(self, gray):
        """
        Apply multiple thresholding methods and choose the best by component count
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        # Otsu thresholding (automatic)
        _, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Mean thresholding (manual fallback)
        mean_val = np.mean(gray)
        _, bw3 = cv2.threshold(gray, mean_val, 255, cv2.THRESH_BINARY_INV)

        # Small denoising pass (reduces speckles)
        for b in (bw1, bw3):
            cv2.medianBlur(b, 3, dst=b)

        # Pick the binary producing a reasonable component count (~12 ideal)
        methods = [bw1, bw3]
        best_bw, best_score = bw1, -1
        for bw in methods:
            k = cv2.connectedComponentsWithStats(bw, 8)[0] - 1
            score = -abs(12 - k)
            if score > best_score:
                best_bw, best_score = bw, score
        return best_bw


    def _filter_components(self, stats, num_labels, image_shape):
        """
        Remove noise, too-small, or too-large regions. Keeps digits and symbols
        """
        if num_labels <= 1:
            return []

        # Gather stats for all components
        areas   = [stats[i, cv2.CC_STAT_AREA]   for i in range(1, num_labels)]
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
        widths  = [stats[i, cv2.CC_STAT_WIDTH]  for i in range(1, num_labels)]
        if not areas:
            return []

        # Compute medians and thresholds
        med_area  = max(1, np.median(areas))
        med_h     = max(1, np.median(heights))
        med_w     = max(1, np.median(widths))
        img_area  = image_shape[0] * image_shape[1]

        min_area_threshold = max(20, self.min_area_ratio * med_area)
        max_area_threshold = 0.35 * img_area  # avoid capturing whole-page blobs
        min_height = max(5, 0.25 * med_h)
        min_width  = max(3, 0.18 * med_w)

        bboxes = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # reject components that are too small or too big
            if area < min_area_threshold or area > max_area_threshold:
                continue

            # aspect ratio classification
            aspect = (w / max(h, 1.0)) if h > 0 else 999.0
            is_minus_like = (h <= 0.40 * med_h) and (aspect >= 4.0)  # likely '-'
            # allow tall-thin like '1' or '/'
            is_tall_thin = False
            if self.keep_tall_thin:
                is_tall_thin = (w <= 0.45 * med_w) and (h >= max(min_height, self.thin_min_height_ratio * med_h)) and (aspect <= 0.45)

            # apply filters
            if not is_minus_like and not is_tall_thin:
                if h < min_height or w < min_width:
                    continue
                ar_sym = max(aspect, 1.0 / max(aspect, 1e-6))
                if ar_sym > self.max_ar:
                    continue

            # ignore tiny border-touching blobs
            H, W = image_shape[:2]
            on_border = (x < 2 or y < 2 or x+w > W-2 or y+h > H-2)
            if on_border and area < med_area * 0.5:
                continue

            bboxes.append((x, y, w, h, area))
        return bboxes


    def _extract_crops(self, bw, bboxes):
        """
        Extract digit crops from binary mask and pad them to square (28x28)
        """
        crops = []
        for (x, y, w, h, _) in bboxes:
            # add small margin
            x0 = max(0, x - self.margin)
            y0 = max(0, y - self.margin)
            x1 = min(bw.shape[1], x + w + self.margin)
            y1 = min(bw.shape[0], y + h + self.margin)
            digit = bw[y0:y1, x0:x1]
            if digit.size == 0:
                continue

            # pad to square shape
            H, W = digit.shape
            s = int(max(H, W) * 1.2)
            pad_t = (s - H) // 2
            pad_b = s - H - pad_t
            pad_l = (s - W) // 2
            pad_r = s - W - pad_l
            sq = cv2.copyMakeBorder(digit, pad_t, pad_b, pad_l, pad_r,
                                    borderType=cv2.BORDER_CONSTANT, value=0)

            # optional centering of the mass
            if self.center:
                ys, xs = np.where(sq > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(xs.mean()), int(ys.mean())
                    tx = sq.shape[1] // 2 - cx
                    ty = sq.shape[0] // 2 - cy
                    tx = np.clip(tx, -sq.shape[1]//4, sq.shape[1]//4)
                    ty = np.clip(ty, -sq.shape[0]//4, sq.shape[0]//4)
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    sq = cv2.warpAffine(sq, M, (sq.shape[1], sq.shape[0]),
                                        flags=cv2.INTER_NEAREST, borderValue=0)

            # resize to fixed output (e.g., 28x28) and re-binarize
            crop = cv2.resize(sq, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
            crop = (crop > 0).astype(np.uint8) * 255
            crops.append(crop)
        return crops


    def _export_objects(self, gray, out_dir="objects_export"):
        os.makedirs(os.path.join(out_dir, "objects"), exist_ok=True)
        
        bw = self.last_bw
        manifest = self.group_manifest

        # -------- save crops using manifest IDs --------
        for comp in manifest["components"]:
            cid = comp["component_id"]
            x, y, w, h = comp["bbox"]
            x0 = max(0, x - 2); y0 = max(0, y - 2)
            x1 = min(bw.shape[1], x + w + 2); y1 = min(bw.shape[0], y + h + 2)
            crop = bw[y0:y1, x0:x1]
            out_path = os.path.join(out_dir, "objects", f"comp_{cid:04}.png")
            cv2.imwrite(out_path, crop)
            comp["crop_path"] = out_path.replace("\\", "/")

        # -------- save manifest --------
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # -------- build reading order (top -> bottom, left -> right) --------
        if "reading_order" in manifest:
            reading_order = manifest["reading_order"]
        else:
            reading_order = sorted(
                range(len(manifest["components"])),
                key=lambda cid: (manifest["components"][cid]["line_idx"],
                                manifest["components"][cid]["pos_in_line"])
            )

        # -------- visualize reading order --------
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        palette = [(0,255,0),(255,0,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0)]

        for ord_idx, cid in enumerate(reading_order):
            comp = manifest["components"][cid]
            x, y, w, h = comp["bbox"]
            color = palette[ord_idx % len(palette)]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label = f"ord:{ord_idx} id:{cid} L{comp['line_idx']}:{comp['pos_in_line']}"
            cv2.putText(vis, label, (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # -------- draw yellow connectors for grouped digits --------
        for g in manifest["groups"]:
            members = g["members"]
            for i in range(len(members) - 1):
                b1 = manifest["components"][members[i]]["bbox"]
                b2 = manifest["components"][members[i + 1]]["bbox"]
                x1, y1, w1, h1 = b1
                x2, y2, w2, h2 = b2
                c1 = (x1 + w1 // 2, y1 + h1 // 2)
                c2 = (x2 + w2 // 2, y2 + h2 // 2)
                cv2.line(vis, c1, c2, (0,255,255), 1)

        cv2.imwrite(os.path.join(out_dir, "debug_reading_order.png"), vis)

    @staticmethod
    def _y_overlap_ratio(b1, b2):
        """
        Compute vertical overlap ratio (IoU) between two boxes
        """
        _, y1, _, h1, _ = b1
        _, y2, _, h2, _ = b2
        a1, b1y = y1, y1 + h1
        a2, b2y = y2, y2 + h2
        inter = max(0, min(b1y, b2y) - max(a1, a2))
        union = (b1y - a1) + (b2y - a2) - inter
        return inter / union if union > 0 else 0.0


    def _group_lines_only(self, bboxes):
        """
        Group bounding boxes into horizontal lines based on vertical overlap and alignment
        """
        if not bboxes:
            return []
        # sort by y (top to bottom), then x
        bboxes = sorted(bboxes, key=lambda b: (b[1], b[0]))
        lines = []
        for b in bboxes:
            placed = False
            cy = b[1] + b[3] * 0.5  # vertical center
            for line in lines:
                # median line height and y-center
                lh  = float(np.median([bb[3] for bb in line]))
                lcy = float(np.median([bb[1] + bb[3]*0.5 for bb in line]))
                # check if roughly same line
                if (self._y_overlap_ratio(b, line[0]) >= self.line_v_overlap) or (abs(cy - lcy) <= lh * self.line_y_tol_ratio):
                    line.append(b); placed = True; break
            if not placed:
                lines.append([b])
        # sort lines top-to-bottom
        lines.sort(key=lambda ln: np.median([bb[1] for bb in ln]))
        # sort each line left-to-right
        for ln in lines:
            ln.sort(key=lambda bb: bb[0] + bb[2] * 0.5)  # use center-x for robustness
        return lines


    def _collect_run_groups(self, line_boxes):
        """
        Within a line, group horizontally-close boxes (like multi-digit numbers)
        """
        if len(line_boxes) <= 1:
            return [line_boxes[:]] if line_boxes else []
        groups, current = [], [line_boxes[0]]
        for nxt in line_boxes[1:]:
            prev = current[-1]
            gap = nxt[0] - (prev[0] + prev[2])  # horizontal gap between boxes
            yov = self._y_overlap_ratio(prev, nxt)
            avg_w = max(3.0, float(np.median([b[2] for b in current] + [nxt[2]])))
            max_gap = self.run_gap_ratio * avg_w
            # merge if close enough horizontally and vertically aligned
            if gap <= max_gap and yov >= self.run_y_overlap:
                current.append(nxt)
            else:
                groups.append(current); current = [nxt]
        groups.append(current)
        return groups


    def _build_group_manifest(self, lines_only, filtered_boxes, img_shape):
        """
        Create manifest with metadata for each component and logical grouping info
        """
        H, W = img_shape[:2]
        atom_id_of = {id(b): i for i, b in enumerate(filtered_boxes)}

        components = []
        groups = []
        group_counter = 0

        # initialize per-component info
        for i, b in enumerate(filtered_boxes):
            components.append({
                "component_id": i,
                "bbox": [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                "line_idx": None,
                "pos_in_line": None,
                "group_id": None,
                "pos_in_group": None
            })

        # assign line positions and build run groups
        for lid, line in enumerate(lines_only):
            for j, b in enumerate(line):
                cid = atom_id_of[id(b)]
                components[cid]["line_idx"] = lid
                components[cid]["pos_in_line"] = j

            runs = self._collect_run_groups(line)
            for run in runs:
                if len(run) >= 2:  # only mark groups with 2+ members
                    member_ids = [atom_id_of[id(b)] for b in run]
                    for k, cid in enumerate(member_ids):
                        components[cid]["group_id"] = group_counter
                        components[cid]["pos_in_group"] = k
                    groups.append({
                        "group_id": group_counter,
                        "line_idx": lid,
                        "members": member_ids
                    })
                    group_counter += 1

        # compute reading order (top→bottom, left→right)
        reading_order = sorted(
            range(len(components)),
            key=lambda cid: (components[cid]["line_idx"], components[cid]["pos_in_line"])
        )
        # add sequential index for convenience
        for k, cid in enumerate(reading_order):
            components[cid]["order_index"] = k

        return {
            "image_size": [int(W), int(H)],
            "components": components,  # all single boxes
            "groups": groups,          # multi-digit runs
            "reading_order": reading_order
        }
