# Nuclei Cell Segmentation Using Transfer Learning

The objective of this project is to segment the cell image and compared the true mask and the prediction mask after model training. The base or pretrained model used for transfer learning is MobileNetV2 and a U-Net model will be construct to restored the size of the image after beng downsampled during feature extractor

## 1. Data loading

In data loading, the inputs image and mask of image was loaded using OpenCV. The image was loaded in RGB format (3 channel) while the masks in grayscale (1 channel). Some of the loaded inputs image and true masks were displayed below.
 
<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/Cell_nuclei_example.PNG" />
</p>

<p align="center">
 The image at the upperside is the cell nuclei image while the second image is it corresponding trues masks of the above image.
</p>

## 2. Data Preprocessing

The input images and true masks were normalized by dividing with 255 which change the range of pixel value (0 to 1).Both image and masks is split using scikitlearn.model_selection method, train_test_split(), and converted into tensor dataset before further split into 'train_dataset' and 'validation_dataset'. The 'train_dataset' and 'validation_dataset' are convert into prefetch.

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/Input_image_and_mask.PNG" />
</p>

<p align="center">
 The inputs image and mask after data preprocessing.
</p>

## 3. Model development

The MobileNetV2 will be use as a feature extraction with frozen layers. The downsampled image will then be upsampled using the U-Net model. The U-Net model structure can be seen below.

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/model.png" />
</p>

## 4. Model evaluation (before and after training)

### Before Training (Training_dataset)

The model were evaluate using train inputs image and true mask to sse the predicted mask before the model training. The ressult can be seen below.

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/Model_evaluate_before_training.PNG" />
</p>

### Training evaluation (Tensorboard)

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/Model_training_acc_loss.PNG" />
</p>

<p align="center">
  From the training graph of accuracy and losses, the graph shows and overfiiting graph. Although the graph is overfit, the gap between the train accuracy and validation accuracy are not to large which is around two percent. The loss also shows the same trend. The accuracy is around 0.96 while the loss is under 0.1.
</p>

### After Training (Validation_dataset)

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/model_evaluate_after_traninig2.PNG" />
</p>

<p align="center">
  The first row predicted mask shows that the predicted mask have a closer match with the true mask. The second and third row prediction mask also show the same results for its coreesponding inputs image.  
</p>

### After Training (Test_dataset)

The test dataset inputs and mask were tested against the train model. The output shows the model manage to predict a correct mask for the test dataset inputs. The image below show several test dataset inputs with its true mask and prediction masks.

<p align="center">
  <img src="https://github.com/acrimn123/Nuclei_Cell_Segmentation/blob/main/PNG/Model_evaluate_test_dataset.PNG" />
</p>

# Acknowledgement

The cell dataset is obtained from [Kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018/overview). 
