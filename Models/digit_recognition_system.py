import os, json
import numpy as np
import tensorflow as tf
from Models.helpers.preprocess import image_preprocessor
from Models.helpers.segmentation import image_segmentation
import cv2
import ast
import operator as op


def _load_model(kind: str, model_path: str | None):
    k = (kind or "cnn").lower()

    if k in ("cnn", "basic", "digits"):
        from Models.CNN import CNN
        from Models.CNN import INV_LABELS as _INV
        model_path = model_path or "./Models/SavedModels/CNN.keras"
        cnn = CNN(model_path)
        return cnn.load_cnn(), _INV

    if k in ("cnn_ext", "extension", "symbols"):
        from Models.CNN_Extension import CNN_Extension as CNN_EXT
        from Models.CNN_Extension import INV_LABELS as _INV
        model_path = model_path or "./Models/SavedModels/CNN_Ext_1.keras"
        cnn_ext = CNN_EXT(model_path)
        return cnn_ext.load_cnn_ext(), _INV

    if k in ("vit", "vt", "transformer"):
        from Models.vt_model import load_vt
        from Models.vt_model import INV_LABELS as _INV
        model_path = model_path or "./Models/SavedModels/vit_ext_2.keras"
        return load_vt(model_path), _INV

    raise ValueError(f"Unknown model kind: {kind}")


OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}

def _evaluate(expr: str):
    def _ev(n):
        if isinstance(n, ast.Num): return n.n
        if isinstance(n, ast.BinOp): return OPS[type(n.op)](_ev(n.left), _ev(n.right))
        if isinstance(n, ast.UnaryOp): return OPS[type(n.op)](_ev(n.operand))
        raise ValueError("Unsupported")
    try:
        return _ev(ast.parse(expr, mode='eval').body)
    except Exception:
        return None


def _pipeline(image_path: str, model, kind: str, inv_labels: dict[int, str], out_dir="digits_export"):
        if model is None:
            raise ValueError("Model not loaded")

        os.makedirs(out_dir, exist_ok=True)

        pre = image_preprocessor(image_path=image_path, binarize=False).preprocess()
        seg = image_segmentation()

        bboxes, crops, manifest = seg.segmentation(pre)
        if not crops:
            payload = {"image_path": image_path, "expression": "", "expression_eval": "", "result": None, "results": []}
            with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return payload

        do_div255 = False
        if kind in ("cnn", "basic", "digits", "cnn_ext", "extension", "symbols"):
            do_div255 = True
            
        processed_crops = []
        for crop in crops:
            # if crop.ndim == 3:
            #     crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # crop = cv2.resize(crop, (W, H))
            crop = crop.astype('float32')
            if do_div255:
                crop /= 255
            crop = crop[..., np.newaxis]
            processed_crops.append(crop)

        X = np.stack(processed_crops, axis=0)

        debug_dir = os.path.join(out_dir, "model_input_preview")
        os.makedirs(debug_dir, exist_ok=True)

        for i, x in enumerate(X):
            # undo the normalization for viewing
            vis = (x[..., 0] * 255).astype("uint8")
            cv2.imwrite(os.path.join(debug_dir, f"input_{i:03}.png"), vis)
        print(f"[DEBUG] Saved {len(X)} preprocessed crops to {debug_dir}")

        preds = model.predict(X, verbose=0)
        probs = tf.nn.softmax(preds).numpy()
        cls_idx = np.argmax(probs, axis=1)
        confs = probs[np.arange(len(probs)), cls_idx]

        labels = [str(inv_labels.get(int(k), str(k))) for k in cls_idx]
     
        ro = manifest.get("reading_order", [c["component_id"] for c in manifest.get("components", [])])
        components = manifest.get("components", [])

        ordered_results = []
        for pos, cid in enumerate(ro):
            comp = components[cid]
            j = comp.get("crop_index", cid)  
            k    = int(cls_idx[j])
            conf = float(confs[j])
            bbox = tuple(map(int, comp["bbox"][:4]))

            ordered_results.append({
                "digit": str(inv_labels.get(k, k)),
                "confidence": conf,
                "bbox": bbox,
                "position": pos,
                "component_id": int(cid),
                "crop_index": int(j),
            })

        expression = "".join(r["digit"] for r in ordered_results)
        eval_map = {"x": "*", "plus": "+", "minus": "-", "slash": "/", "equals": "="}
        eval_expr = "".join(eval_map.get(r["digit"], r["digit"]) for r in ordered_results)

        allowed = set("0123456789+-*/()")
        can_eval = all(ch in allowed for ch in eval_expr)
        result = _evaluate(eval_expr) if can_eval and eval_expr else None
        
        payload = {
            "image_path": image_path,
            "expression": expression,
            "expression_eval": eval_expr,
            "result": result,
            "results": ordered_results
        }

        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent =2)

        return payload


def run(image_path: str, kind: str = "cnn", model_path: str | None = None, out_dir: str = "digits_export"):
    model, inv_label = _load_model(kind, model_path)
    # print("Loaded: ", type(model) )
    return _pipeline(image_path, model, kind, inv_label, out_dir)