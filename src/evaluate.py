import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to load the dataset
def load_dataset(data_dir, batch_size=64, num_workers=2):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    return dataloader, dataset_size, class_names

# Function to load the trained model
def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to evaluate the model
def evaluate_model(model, dataloader, device, class_names):
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1])

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_eval.png')  # Save confusion matrix
    plt.show()

    return accuracy, precision, recall, f1, auc, cm

# Main function
def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data directory
    data_dir = 'data/chest_xray'  # Update this to wherever your dataset is located
    model_path = 'model/pneumonia_detection_model.pth'  # Update this to your trained model path
    num_classes = 2

    # Load the dataset
    dataloader, dataset_size, class_names = load_dataset(data_dir)

    # Load the model
    model = load_model(model_path, num_classes)
    model = model.to(device)

    # Evaluate the model
    evaluate_model(model, dataloader, device, class_names)

if __name__ == "__main__":
    main()
