# Unsupervised Anomaly Detection Segmentation Approach with Anomalib


Anomalib is a Deep learning library Collection of the state of the art anomaly detection algorithms. 
The Task was assigned to me during an interview process several images of fabrics with defects were given and was asked to perform unsupervised anomaly detection with segmentation approach. Ensure that your environment is CUDA enabled.

refer to notebook **Anomalib_fabric_fault_detection_segmentation_fastflow.ipynb** to walk through the code.


## Project Overview ##

Anomalib is a deep learning library that collects state-of-the-art anomaly detection algorithms for benchmarking on various datasets. For this project, I used Anomalib to perform unsupervised anomaly detection on fabric images. The task involved detecting defects in fabric images using a segmentation approach.

We're using **Anomalib==0.7.0** for this project.


## Dataset ##
The dataset used in this project consists of images of fabrics with defects. The defects were annotated using the **Roboflow** annotation tool [click here](https://roboflow.com/annotate), and the dataset was downloaded in COCO format. Binary masks were generated for each image using a custom Python script.



## Model ##
The FastFlow model was used for training. FastFlow is a deep learning model designed for anomaly detection. For more details about the FastFlow model with **resnet15** backbone, please refer to the FastFlow published article [link](https://arxiv.org/abs/2111.07677).


## Training ##
The model was trained using the generated binary masks and defect images. The training process was managed using Weights & Biases for logging and monitoring.

## Installation ##

To set up the environment for this project, follow these steps:

1. ## Clone the Repository: ##

   ```bash
   git clone https://github.com/KaustubhK94/Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib.git
   cd Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib
   ```
2. ## Create A Virtual Environment(Optional): ##
   ```bash
   python -m venv venv
   ```
3. ## Install Dependencies: ##
   ```bash
   pip install -r requirements.txt
   ```   



## Running Predictions ##

To run predictions on an image using the `predict.py` script, use the following command:

```bash
python predict.py --image "/path/to/image.jpg" --weights "/path/to/model_weights.ckpt"
```


## Sample Binary Mask ##

Below are two sample binary mask images used in this project:

<table>
  <tr>
    <td>
      <img src="/Fabric_defect/Fabric22.jpg" alt="Fabric Defect" width="400"/>
    </td>
    <td>
      <img src="/Binary_masks/Fabric22.jpg" alt="Binary Mask" width="400"/>
    </td>
  </tr>
</table>



Images with Defects
Here is a sample image from the dataset:


## Files

- **Images with defects:** [Fabric_defect/](./Fabric_defect/)
- **Binary masks:** [Binary_masks/](./Binary_masks/)
- **Dataset in COCO format:** [Fabric_defects.v4i.coco-segmentation/](./Fabric_defects.v4i.coco-segmentation/)
- **Trained model weights:** [best-model-v1.ckpt](./best-model-v1.ckpt)

![](/Fabric20.jpg)

## Weights & Biases Plots

Below are some W&B plots from the project:

<table>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/KaustubhK94/Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib/master/Metrics%20Monitered/W&B%20Chart%209_8_2024,%202_59_45%20PM.png" width="400"/>
      <p><strong>Plot 1:</strong> Train Loss Epoch</p>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/KaustubhK94/Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib/master/Metrics%20Monitered/W&B%20Chart%209_8_2024,%202_59_56%20PM.png" width="400"/>
      <p><strong>Plot 2:</strong> PIXEL_AUROC</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/KaustubhK94/Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib/master/Metrics%20Monitered/W&B%20Chart%209_8_2024,%203_00_16%20PM.png" width="400"/>
      <p><strong>Plot 3:</strong> Image_AUROC</p>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/KaustubhK94/Unsupervised_Anomaly_Detection_segmentation_approach_Anomalib/master/Metrics%20Monitered/W&B%20Chart%209_8_2024,%203_00_04%20PM.png" width="400"/>
      <p><strong>Plot 4:</strong> Epochs</p>
    </td>
  </tr>
</table>




