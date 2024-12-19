# Image Classification of Fish Species Using the CIFAR-100 Dataset  

This repository showcases a deep learning project aimed at classifying fish species from the CIFAR-100 dataset. Leveraging the power of transfer learning with the VGG16 model, this implementation accurately categorizes images into five fine-grained fish species classes: aquarium fish, flatfish, ray, shark, and trout.

## Features  
- **Dataset**: CIFAR-100, filtered for fish species classification.  
- **Model Architecture**: Transfer learning using VGG16 with custom classification layers.  
- **Training Optimization**: Dropout regularization and learning rate scheduling.  
- **Evaluation Tools**: Confusion matrix and accuracy visualization.  
- **Visualization**: Training history and prediction results.

## Prerequisites  
Ensure you have the following installed:  
- Python 3.7 or higher  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- OpenCV  

Install all dependencies using:  
```bash
pip install tensorflow keras numpy matplotlib opencv-python
```
## Dataset  
The CIFAR-100 dataset comprises 60,000 32x32 color images across 100 classes. For this project, only the following fish species classes were selected:  
- Aquarium Fish  
- Flatfish  
- Ray  
- Shark  
- Trout  

The dataset was filtered to include only these classes, and the images were resized to 224x224 to match the input requirements of the VGG16 model.

## Project Structure
```bash
├── Source Code/
│   └── train_fish_cifar100.py   # Script for training and evaluating the model
├── README.md                   # Documentation
├── data/                       # Directory for dataset (to be created by the user)
├── results/                    # Directory for storing output (e.g., visualizations, metrics)
```
## Model Architecture  
The project utilizes a modified VGG16 model with the following components:  
- **Pre-trained Layers**: The VGG16 model, pre-trained on ImageNet, is used as the feature extractor with its convolutional layers frozen to retain pre-learned weights.  
- **Custom Fully Connected Layers**: Added on top of the pre-trained layers to handle the fish species classification task.  
- **Dropout Layers**: Used to reduce overfitting during training by randomly disabling neurons.  
- **Softmax Activation**: Ensures probabilistic outputs for multi-class classification.

## Training Configuration  
- **Loss Function**: Sparse categorical cross-entropy is used to handle integer-encoded class labels.  
- **Optimizer**: Adam optimizer with learning rate scheduling to adapt learning rates dynamically.  
- **Epochs**: The model is trained for 20 epochs.  
- **Input Size**: Images are resized from 32x32 to 224x224 to match the VGG16 input size requirements.

## How to Run  
1. Clone the repository:  
```bash
git clone https://github.com/Gengo-bit/Image-Classification-of-Fish-Species-Using-the-CIFAR-100-Dataset
cd Image-Classification-of-Fish-Species-Using-the-CIFAR-100-Dataset
```
2. Prepare the CIFAR-100 dataset by downloading it and organizing the files into a data/ directory.
3. Train the model by running the script:
   ```bash
   cd Source Code
   python train_fish_cifar100.py
4.  The results, including training metrics and visualizations, will be saved in the results/ directory.

## Results  
The model demonstrated an accuracy range of **68% to 76%**.

### Key Insights  
- **Training and Validation Loss Trends**:  
  - Training loss consistently decreases.  
  - Validation loss stagnates or increases after ~12 epochs, indicating overfitting.  
- **Confusion Matrix**:  
  - Highlights the model's classification performance and areas of misclassification.  
- **Visualization**:  
  - Displays predictions with true vs. predicted labels for test images.

## Challenges & Solutions  
- **Overfitting**:  
  - **Challenge**: The model overfits after a few epochs.  
  - **Solution**: Dropout layers and reduced model complexity were implemented to mitigate this issue.  
- **Long Training Times**:  
  - **Challenge**: The large dataset and extended epochs increased training time.  
  - **Solution**: Learning rate scheduling was applied to speed up convergence.  
- **Fluctuating Accuracy**:  
  - **Challenge**: Accuracy during training was inconsistent.  
  - **Solution**: Hyperparameter tuning and reduced epochs improved stability.

## Authors  
- Paul Emmanuel Corsino  
- David Emmanuel Lacsao  
- Klinnsonveins Yee  

## License  
This project is licensed under the MIT License. See the LICENSE file for details.

## References  
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)  
- [VGG16 Model Documentation](https://keras.io/api/applications/vgg/#vgg16-function)  
