import torch
from torchvision import transforms

from src.model import CNN

MNIST_CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
EMNIST_LETTER_CLASS_NAMES = [chr(ord("A") + index) for index in range(26)]

def get_class_names(dataset_name="mnist"):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return MNIST_CLASS_NAMES
    if dataset_name in {"emnist", "emnist_letters"}:
        return EMNIST_LETTER_CLASS_NAMES
    raise ValueError("dataset_name must be 'mnist' or 'emnist_letters'")

def load_model(model_path="models/mnist_cnn.pth", dataset_name="mnist"):
    class_names = get_class_names(dataset_name)
    model = CNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    return transform(image).unsqueeze(0)

def predict_image(image, model=None, dataset_name="mnist"):
    if model is None:
        model = load_model(dataset_name=dataset_name)

    class_names = get_class_names(dataset_name)
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    label = class_names[predicted_class.item()]
    confidence = confidence.item()

    return label, confidence

def predict_tensor(
    tensor,
    model,
    class_names
):

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(
            output,
            dim=1
        )
        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    label = class_names[
        predicted_class.item()
    ]

    confidence = confidence.item()
    return label, confidence


# if __name__ == "__main__":
#     model = load_model()

#     image_path = input("Enter image path: ")
#     image = Image.open(image_path)

#     label, confidence = predict_image(image, model)

#     print(f"Prediction: {label}")
#     print(f"Confidence: {confidence * 100:.2f}%")
