#%%
#Import packages
import tensorflow as tf
from tensorflow import keras
from keras import layers,losses,optimizers,callbacks
from tensorflow_examples.models.pix2pix import pix2pix
from keras.callbacks import TensorBoard
from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras.metrics import IoU, Accuracy
import datetime
import cv2
import numpy as np
import os

#1. Load train data

#1.1. Prepare an empty list for the images and masks
images = []
masks = []
root_path = os.path.join(os.getcwd(),'dataset','train')

#1.2. Load the images using opencv
image_dir = os.path.join(root_path,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)
    
#1.3. Load the masks
mask_dir = os.path.join(root_path,'masks')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)
    
#%%
#2. Visualize data 
#1.4. Convert the lists into numpy array
images_np = np.array(images)
masks_np = np.array(masks)


# Check some examples
plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_np[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize=(10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_np[i])
    plt.axis('off')
    
plt.show()

#%%
#3. Data preprocessing

# Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np,axis=-1)

#Check the mask output
print(np.unique(masks[0]))

#%%
# Convert the mask values into class labels
converted_masks = np.round(masks_np_exp/255).astype(np.int64)

#Check the mask output
print(np.unique(converted_masks[0]))

#%%
# Normalize image pixels value
converted_images = images_np / 255.0
sample = converted_images[0]

#%%
# Perform train-test split
from sklearn.model_selection import train_test_split

SEED = 12345
x_train,x_test,y_train,y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

#%%
#4. Convert the numpy arrays into tensor 
x_train_tensor = tf.data.Dataset.from_tensor_slices(x_train)
x_test_tensor = tf.data.Dataset.from_tensor_slices(x_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#5. Separate into train dataset
train_dataset = tf.data.Dataset.zip((x_train_tensor,y_train_tensor))
val_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

#%%
#6. Data augmentation function
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#%%
#7. Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().map(Augment()).prefetch(buffer_size=tf.data.AUTOTUNE))
val_batches = val_dataset.batch(BATCH_SIZE)

#%%
# Visualize some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    
#%%
#8. Create image segmentation model

#8.1. Use a pretrained model as the feature extraction layers
base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#8.2. List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%%
# Make of use of the function to construct the entire U-Net
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

#Compile the model
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
keras.utils.plot_model(model, show_shapes=True)

#%%
#Create functions to show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None,num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

#Test out the show_prediction function
show_predictions()

#%%
#Create a callback to help display results during model training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))

# tensorboard callbacks
log_path=os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb=callbacks.TensorBoard(log_dir=log_path)

#%%
# 9. Model training
#Hyperparameters for the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches,validation_data=val_batches,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,callbacks=[DisplayCallback(),tb])

#%%
# 10. Model Evaluation
show_predictions(val_batches,3)

#%%
#11. Load test data

#11.1 Empty list for test inputs and masks
test_images = []
test_masks = []
test_path = os.path.join(os.getcwd(),'dataset','test')

#11.2. Load the test images using opencv
tst_dir = os.path.join(test_path,'inputs')
for tst_image in os.listdir(tst_dir):
    img_ = cv2.imread(os.path.join(tst_dir,tst_image))
    img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
    img_ = cv2.resize(img_,(128,128))
    test_images.append(img_)
    
#11.3. Load the test masks
tst_dir_ = os.path.join(test_path,'masks')
for tst_mask in os.listdir(tst_dir_):
    mask_ = cv2.imread(os.path.join(tst_dir_,tst_mask),cv2.IMREAD_GRAYSCALE)
    mask_ = cv2.resize(mask_,(128,128))
    test_masks.append(mask_)

#%%
#12. Test Data preparation

# Convert the lists into numpy array
images_np_test = np.array(test_images)
masks_np_test = np.array(test_masks)

# Expand the mask dimension
masks_np_exp_test = np.expand_dims(masks_np_test,axis=-1)

# Convert the mask values into class labels
converted_masks_test = np.round(masks_np_exp_test/255).astype(np.int64)

#Check the mask output
print(np.unique(converted_masks_test[0]))

# Normalize image pixels value
converted_images_test = images_np_test/ 255.0
sample = converted_images_test[0]

# convert to tensor
test_input_tensor = tf.data.Dataset.from_tensor_slices(converted_images_test)
test_mask_tensor = tf.data.Dataset.from_tensor_slices(converted_masks_test )

# zip test
test_dataset = tf.data.Dataset.zip((test_input_tensor,test_mask_tensor))

# test batches
test_batches = test_dataset.batch(BATCH_SIZE)

#%%
# 13. Show prediction for test batches
show_predictions(test_batches,3)

#%%
#14. Model save
save_path = os.path.join(os.getcwd(),'Cell_nuclei_segmentation.h5')
model.save(save_path)

# %%
