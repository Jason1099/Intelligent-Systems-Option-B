import os
from digit_recognition_system import DigitRecognitionSystem

if __name__ == "__main__":
    drs = DigitRecognitionSystem()
    model_path = 'vit_mnist_model.keras'

    # Load the model or use the existing one
    if os.path.exists(model_path):
        drs.load_model(model_path)
    else:
        drs.train_model(epochs=100, batch_size=256, vanilla=False)
        drs.save_model(model_path)

    # Run prediction on the given test image
    test_image = 'handWrittenDigitsTest.png'
    if os.path.exists(test_image):
        results = drs.predict_image(test_image)
        
        print("\nPrediction Results:")
        for result in results:
            print(f"Position {result['position']}: Digit {result['digit']} "
                  f"(Confidence: {result['confidence']:.2%})")