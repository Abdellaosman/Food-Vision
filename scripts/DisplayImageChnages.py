import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to apply preprocessing steps step-by-step
def show_preprocessing_steps(image_path):
    # Load the original image
    original_image = Image.open(image_path).convert('RGB')

    # Define individual preprocessing steps (matching your training pipeline)
    steps = [
        ("Original Image", None),
        ("Resized Image", transforms.Resize((128, 128))),
        ("Random Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),  # Always flip for demonstration
        ("Random Rotation", transforms.RandomRotation(30)),
        ("Color Jitter", transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)),
        ("Random Affine", transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2))),
        ("Random Perspective", transforms.RandomPerspective(distortion_scale=0.5, p=1.0)),  # Always distort for demo
        ("Normalized Image", transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
    ]

    # Display each transformation step
    images = [(original_image, "Original Image")]  # Add original image first
    current_image = original_image
    for title, transform in steps[1:]:
        if transform:
            current_image = transform(current_image)
        images.append((current_image, title))

    # Display all transformations
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(img, torch.Tensor):  # Handle normalized image
            # Unnormalize and convert back to PIL image for visualization
            unnormalize = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
            img = unnormalize(img)
            img = torch.clamp(img, 0, 1)
            img = transforms.ToPILImage()(img)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with the path to your image
    image_path = "C:/Users/Abdella/OneDrive/Desktop/test/Food-Vision/sample_pics/22.jpg"
    show_preprocessing_steps(image_path)
