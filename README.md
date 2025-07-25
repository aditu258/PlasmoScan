# Malaria Cell Detection

Detect malaria-infected and uninfected cells using deep learning (custom CNN and transfer learning with VGG16).

## Project Overview
This project classifies cell images as **Parasitized** (malaria-infected) or **Uninfected** using:
- A custom Convolutional Neural Network (CNN)
- Transfer learning with VGG16 (feature extraction and fine-tuning)

## Workflow
1. **Dataset Download**
   - Download the dataset from [Kaggle: cell-images-for-detecting-malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
   - Unzip to `cell_images/` with subfolders `Parasitized/` and `Uninfected/`
2. **Data Preparation**
   - Images are loaded and preprocessed using `ImageDataGenerator` with augmentation (rescale, zoom, flip)
   - 80/20 train/validation split
3. **Modeling Approaches**
   - **Custom CNN:**
     - 3 Conv2D blocks (with BatchNorm, MaxPooling, Dropout)
     - Dense layers for binary classification
     - Trained for 10 epochs (Adam optimizer, learning rate 0.0003)
   - **Transfer Learning (VGG16):**
     - VGG16 base (pretrained on ImageNet, no top)
     - Custom dense layers on top
     - **Feature Extraction:** VGG16 base is frozen, only new layers are trained
     - **Fine-Tuning:** Last convolutional block of VGG16 is unfrozen and trained along with the new layers for improved accuracy
     - Trained for 10 epochs (RMSprop optimizer, learning rate 1e-5 for fine-tuning)
4. **Evaluation**
   - Training and validation accuracy/loss plotted for each approach
   - Typical results: >96% validation accuracy after fine-tuning
5. **Model Saving**
   - Models saved as `.h5` files (e.g., `malaria_model.h5`, `malaria_model_finetuning.h5`)
6. **Prediction & Visualization**
   - Random images from both classes are predicted and visualized with confidence scores

## Fine-Tuning (VGG16)
Fine-tuning is a crucial step that allows the model to adapt the deeper layers of VGG16 to the malaria cell dataset:
- **Feature Extraction:** Initially, all VGG16 layers are frozen and only the custom classifier is trained.
- **Fine-Tuning:** The last convolutional block of VGG16 is unfrozen, allowing its weights to be updated during training. This helps the model learn dataset-specific features and improves accuracy.
- **How to Run:**
  - In the notebook, follow the section labeled `Fine Tuning`.
  - The code will unfreeze the last block, recompile the model with a low learning rate, and continue training.
- **Results:**
  - Fine-tuning typically increases validation accuracy to ~96% or higher.
  - The notebook provides plots to compare training/validation accuracy and loss before and after fine-tuning.

## Usage
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download and extract the dataset** from Kaggle as described above.
3. **Run the notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```
   - Follow the notebook cells to train, evaluate, and visualize predictions.
   - For fine-tuning, execute the cells in the `Fine Tuning` section after initial transfer learning training.

## Docker
A Dockerfile is provided for containerized deployment:

1. **Build the Docker image**:
   ```bash
   docker build -t malaria-cell-detection .
   ```
2. **Run the container**:
   ```bash
   docker run -p 8888:8888 -v $(pwd)/cell_images:/app/cell_images malaria-cell-detection
   ```
   - Access Jupyter at `http://localhost:8888`
   - Mount the dataset as a volume for large data

## Deployment
- **Local:** Use Docker or run the notebook directly
- **Cloud:** Deploy the Docker image to any cloud provider supporting containers (AWS, GCP, Azure, etc.)
- **Note:** Ensure the dataset is available in the container or mounted as a volume

## Requirements
- Python 3.8+
- See `requirements.txt` for all Python dependencies:
  - tensorflow
  - numpy
  - matplotlib
  - scikit-learn
  - pillow
  - opencv-python
  - tqdm

## Results
- **Custom CNN:** ~93% validation accuracy
- **VGG16 Transfer Learning:** Up to ~96% validation accuracy after fine-tuning
- **Visualization:** The notebook includes code to visualize predictions and model performance
- **Fine-Tuning:** Provides a clear boost in accuracy and generalization for the malaria cell dataset

## License
MIT License

---
**Keywords:** Docker, Deployment, Malaria, Deep Learning, TensorFlow, VGG16, Transfer Learning, Fine-Tuning, Cell Classification 