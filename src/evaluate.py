import argparse
import torch
import torch.nn as nn

from src.model import CNN
from src.dataset import loaders
from predict import get_class_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "emnist_letters"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    data_loaders = loaders(batch_size=args.batch_size, dataset_name=args.dataset)
    class_names = get_class_names(args.dataset)
    
    model = CNN(num_classes=len(class_names))
    model_path = args.model_path or (
        "models/emnist_letters_cnn.pth"
        if args.dataset == "emnist_letters"
        else "models/mnist_cnn.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loaders["test"]:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    total = len(data_loaders["test"].dataset)
    print(f"Test Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")

if __name__ == "__main__":
    main()
