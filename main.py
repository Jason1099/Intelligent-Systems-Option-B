import os
from Models.digit_recognition_system import run

if __name__ == "__main__":
    # Choose which model to use: "cnn", "cnn_ext", or "vit"
    # model_kind = "cnn"
    model_kind = "cnn_ext"
    # model_kind = "vit"

    dir = "./Input_Images/"

    # Chosse the input image
    image_name = "image.png"

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
        print(f"Expression: {results['expression_eval']}")
        print(f"Evaluated Result: {results['result']}")
        print("\nDetailed Components:")
        for r in results["results"]:
            conf = r.get("confidence", 0)
            print(f"  Pos {r['position']}: '{r['digit']}' (Confidence: {conf:.2%})")

        print("\nResults saved to digits_export/results.json")
