## Vehicle Color Recognition
This project aims to build a computer vision model that classifies the color of vehicles from images. I leverage a convolutional neural network (CNN) to extract visual features and predict one of several color categories (e.g. red, blue, black, etc.). Convolutional networks have proven effective for image classification tasks. In our setup, I fine-tune a pretrained ResNet-18 model to recognize vehicle colors automatically.

## Dataset

I use the VCoR (Vehicle Color Recognition) dataset, a publicly available collection on Kaggle. This dataset contains over 10,000 labeled vehicle images spanning 15 color categories. The colors include common vehicle hues such as white, black, gray, silver, red, blue, brown, green, beige, orange, gold, yellow, purple, pink, and tan. The data is organized into training, validation, and test splits, with images grouped by color label. (See the VCoR Kaggle page for details.)

## Technologies Used

- Python 3 and PyTorch: for model development and training.
- torchvision: provides datasets, image transforms, and pretrained models.
- ResNet-18: a standard CNN architecture (pretrained on ImageNet) used as the backbone.
- NumPy and Pandas: for data manipulation.
- Matplotlib (and seaborn): for plotting training curves and the confusion matrix.
- Jupyter Notebook: the code is provided as an executable notebook.

## Model Architecture
I adopt a pretrained ResNet-18 network as our feature extractor. ResNet-18 is an 18-layer deep convolutional neural network trained on over a million images (ImageNet). I replace the final fully-connected layer to output 15 classes (one per color) and then fine-tune the network on our VCoR images. This transfer learning approach (pretraining on a large dataset and fine-tuning for a specific task) is a common practice in image classification. Using a pretrained model allows the network to start with rich feature representations and quickly adapt to vehicle colors in our dataset.

## Results
The trained model achieves a well-enough performance on the vehicle color classification task. For example, the final validation accuracy (after fine-tuning) is 87%. I present a confusion matrix showing the model’s predictions versus true color labels; most colors (e.g. red, blue, black, white) are classified correctly, while some similar shades (e.g. beige vs tan or gray vs silver) show higher confusion. The notebook includes visual examples of correct and incorrect predictions to illustrate model behavior. Overall, the results demonstrate that our CNN can reliably identify vehicle colors from images.

## Model Weights
After training, the final model weights are saved to disk (e.g. as vehicle_color_model.pth). This serialized file contains the learned parameters of the ResNet-18 network after fine-tuning. Users can load this file in Python to perform inference without retraining. In the notebook, the torch.save() function is used at the end of training to dump the model weights.
