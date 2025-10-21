import os, cv2, json, numpy as np
from preprocess import image_preprocessor
from segmentation import image_segmentation

# -------- settings --------
IMAGE_PATH = "testImage.png"     
OUT_DIR    = "digits_export"
os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)

# -------- load & preprocess --------
processor = image_preprocessor(IMAGE_PATH, size=(28, 28))
pre = processor.preprocess()          
# processor.show_image(pre)             

# -------- segmentation --------
seg = image_segmentation(
    run_gap_ratio=0.35, 
    run_y_overlap=0.45,
    line_y_tol_ratio=0.35,
    keep_tall_thin=True,
    thicken_ones=True
)
atoms, _ = seg.segmentation(pre)
manifest = seg.group_manifest
bw = seg.last_bw

# -------- save crops using manifest IDs --------
for comp in manifest["components"]:
    cid = comp["component_id"]
    x, y, w, h = comp["bbox"]
    x0 = max(0, x - 2); y0 = max(0, y - 2)
    x1 = min(bw.shape[1], x + w + 2); y1 = min(bw.shape[0], y + h + 2)
    crop = bw[y0:y1, x0:x1]
    out_path = os.path.join(OUT_DIR, "train", f"comp_{cid:04}.png")
    cv2.imwrite(out_path, crop)
    comp["crop_path"] = out_path.replace("\\", "/")

# -------- save manifest --------
with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"Saved manifest with {len(manifest['components'])} components and "
      f"{len(manifest['groups'])} groups.")

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
vis = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
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

cv2.imwrite(os.path.join(OUT_DIR, "debug_reading_order.png"), vis)
print(f"Saved debug_reading_order.png showing true left -> right order.")

# -------- Table to verify ordering --------
print("\n=== Verified Left→Right Per Line ===")
lines = {}
for comp in manifest["components"]:
    lid = comp["line_idx"]
    lines.setdefault(lid, []).append(comp)

for lid, comps in sorted(lines.items()):
    comps.sort(key=lambda c: c["pos_in_line"])
    ids = [c["component_id"] for c in comps]
    xs = [c["bbox"][0] for c in comps]
    print(f"Line {lid}: IDs {ids}  Xs {xs}")
