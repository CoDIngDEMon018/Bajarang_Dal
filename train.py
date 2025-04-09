import os
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import argparse

# Clear GPU memory
torch.cuda.empty_cache()

# Hyperparameters
EPOCHS = 20  # Reduced number of epochs for faster training
MOSAIC = 0.75
OPTIMIZER = 'AdamW'
MOMENTUM = 0.2
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False

# Custom Dataset Class
class CustomYOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Load corresponding label
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Convert labels to tensor
        labels = [list(map(float, line.strip().split())) for line in labels]
        labels = torch.tensor(labels)

        return image, labels

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Reduced image size to 640x640
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()

    # Dataset and DataLoader
    image_dir = "E:/projects/MLH/HackByte_Dataset/data/train/images"
    label_dir = "E:/projects/MLH/HackByte_Dataset/data/train/labels"
    dataset = CustomYOLODataset(image_dir, label_dir, img_size=640, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=None, num_workers=2)  # Reduced batch size

    # Initialize YOLO model
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # Freeze only the initial layers
    freeze_count =20   # Number of initial layers to freeze
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < freeze_count:  # Freeze the first `freeze_count` layers
            param.requires_grad = False

    # Train the model using the built-in train method
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),  # Path to dataset configuration
        epochs=args.epochs,
        imgsz=1024,  # Reduced image size
        batch=8,  # Reduced batch size
        lr0=args.lr0,  # Initial learning rate
        lrf=args.lrf,  # Final learning rate
        optimizer=args.optimizer,  # Optimizer
        weight_decay=0.05,  # Regularization
        mosaic=args.mosaic,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation
        single_cls=args.single_cls,  # Single class training
        patience=10,  # Early stopping after 10 epochs of no improvement
        workers=2,  # Number of workers for DataLoader
        amp=True  # Enable mixed precision training
    )

    print("Training completed successfully!")
    print("Results:", results)
