import os
from Models.digit_recognition_system import run

def main(model_kind : str = "cnn", image_name : str = ""):
    # Choose which model to use: "cnn", "cnn_ext", "vit" or "vit_ext"
    # model_kind = "cnn"
    # model_kind = "cnn_ext"
    # model_kind = "vit"
    # model_kind = "vit_ext"

    dir = "./Input_Images/"

    # Chosse the input image
    test_image = dir + image_name

    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
    else:
        print(f"Running Digit Recognition using '{model_kind}' model...")
        results = run(
            image_path=test_image,
            kind=model_kind,
            out_dir="digits_export"
        )

        print("\n Prediction Complete!")

        line_results = results.get("line_results")
        digit_results = results.get("digit_results")

        lines = ""
        print("\nPer-line expressions:")
        for lr in sorted(line_results, key=lambda x: x.get("line_index", 0)):
            li   = lr.get("line_index", 0)
            expr = lr.get("expression", "")
            val  = lr.get("result", None)

            lines += f"line {li}: {expr}  =>  {val}"

            print(f"  line {li}: {expr}  =>  {val}")

        print("\nDetailed Components:")
        rows = digit_results
        pred = ""
        for r in rows:
            pos   = r.get("position", "?")
            dig   = r.get("digit", "?")
            conf  = r.get("confidence", 0.0)
            li    = r.get("line_index", r.get("line_idx", "?"))
            bbox  = r.get("bbox", None)

            pred = f"  pos {pos} | line {li} | '{dig}' | conf {conf:.2%}"

            print(f"  pos {pos} | line {li} | '{dig}' | conf {conf:.2%}" + (f" | bbox {bbox}" if bbox else ""))

        print("\nResults saved to digits_export/results.json")

        return line_results, digit_results, lines, pred
    

_, _, lines, pred  = main(model_kind = "cnn", image_name = "handWrittenDigitsTest.png")