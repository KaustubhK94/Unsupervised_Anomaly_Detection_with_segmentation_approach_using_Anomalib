from pathlib import Path
import torch
from pytorch_lightning import Trainer
from anomalib.models.fastflow.lightning_model import Fastflow
from anomalib.data import InferenceDataset
from torch.utils.data import DataLoader
from anomalib.post_processing import superimpose_anomaly_map
from anomalib.pre_processing.transforms import Denormalize
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# Function to resize the input image if necessary
def resize_image(image_path, image_size):
    image = Image.open(image_path)
    if image.size != image_size:
        print(f"Resizing image from {image.size} to {image_size}")
        image = image.resize(image_size)
    return image

# Function to perform inference
def perform_inference(image_path, model_weights, image_size=(256, 256), flow_steps=8, backbone="resnet18"):
    # Initialize the model with the same configuration as used during training
    model = Fastflow(input_size=image_size, backbone=backbone, flow_steps=flow_steps)

    # Load the saved weights
    checkpoint = torch.load(model_weights, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    # Initialize Trainer without WandB logger
    trainer = Trainer(
        callbacks=[],
        accelerator="auto",
        devices=1
    )

    # Resize the image if it doesn't match the model's expected input size
    resized_image = resize_image(image_path, image_size)

    # Save the resized image temporarily
    resized_image_path = Path("/tmp/resized_image.jpg")
    resized_image.save(resized_image_path)

    # Prepare your inference dataset and dataloader
    inference_dataset = InferenceDataset(path=resized_image_path, image_size=image_size)
    inference_dataloader = DataLoader(dataset=inference_dataset)

    # Perform inference
    predictions = trainer.predict(model=model, dataloaders=inference_dataloader)[0]

    # Extract and process predictions
    image = predictions["image"][0]
    image = Denormalize()(image)

    anomaly_map = predictions["anomaly_maps"][0]
    anomaly_map = anomaly_map.cpu().numpy().squeeze()

    # Display the image, anomaly map, and heatmap
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title("Anomaly Map")
    plt.imshow(anomaly_map, cmap="jet")

    heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
    plt.subplot(1, 3, 3)
    plt.title("Heatmap")
    plt.imshow(heat_map)

    plt.show()

    # Print additional prediction results
    print("Prediction Results:")
    pred_score = predictions["pred_scores"][0]
    pred_labels = predictions["pred_labels"][0]
    pred_masks = predictions["pred_masks"][0].squeeze().cpu().numpy()

    print(f"Prediction Score: {pred_score}")
    print(f"Predicted Label: {pred_labels}")
    
    # Display predicted mask
    plt.figure()
    plt.title("Predicted Mask")
    plt.imshow(pred_masks)
    plt.show()

    # Show additional image details for debugging
    print(f"Image Shape: {image.shape}\nMin Pixel: {image.min()}\nMax Pixel: {image.max()}")
    print(f"Anomaly Map Shape: {anomaly_map.shape}")
    print(f"Predicted Mask Shape: {pred_masks.shape}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Perform inference on an image using the trained FastFlow model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file for inference.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the saved model weights.")
    args = parser.parse_args()

    # Call the inference function
    perform_inference(image_path=args.image, model_weights=args.weights)
