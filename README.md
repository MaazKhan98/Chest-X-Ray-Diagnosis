# CHEST X-ray Diagnosis
This is the project of Chest X-Ray classifier that distinguishes between COVID-19 infected lungs, Viral Pneumonia and healthy lungs using several deep learning architectures and compare their efficiency to the mentioned task. 
I have used 7 different neural networks to analyze performance and highlight using GRADCAM heatmaps for convolutional neural networks.

## Dataset
Data is imported from Kaggle Library, which contains chest X-ray images for COVID-19 positive cases along with Viral Pneumonia and normal cases images. It consists of 1200 positive COVID-19 images, 1345 viral pneumonia images, and 1341 normal images.
The database is splitted randomly into three folder: Training, Validation, and Testing, as 80%, 10%, and 10% respectively. Each folder consists of three sub-folders; Covid-19, Pneumonia, and Normal.

## Data preprocessing:
Before training the neural network, all the images are standarized first, by calling the Normalization layer from keras and is adapted on the training set only to remain unbiased. The standarization is done on pixel level as every pixel is one dimension so we get the mean and standard deviation of each position of the pixel and then then every pixel in the input image is standarized so that the total image has a mean of zero and standard deviation equals 1.

## Workflow
The first step pre-processes the data and creates data pipelines, then explores several deep learning architectures beginning with a simple shallow fully connected neural network and another deep fully connected neural network then passing by a shallow convolutional neural network and a deep convolutional neural network and finally ending with famous convolutional neural networks such as: ResNet50, DenseNet121 and InceptionV3.
The last step is to apply gradient class activation mappings (GRAD-CAM) to segment the areas that mostly affected the decision of the model. GRAD-CAM is only applied to convolutional neural networks.

## Results
All models are trained with batch size of 32 for training and 16 for testing over 100 epochs. 
The following tables compares the models after a maximum of 100 epochs:
https://github.com/MaazKhan98/Chest-X-ray-Diagnosis/blob/master/Results.png
