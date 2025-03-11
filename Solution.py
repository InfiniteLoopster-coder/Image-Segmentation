import datetime
import os
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Set up matplotlib colormap
plt.rcParams['image.cmap'] = 'Greys_r'

# Global parameters
IMG_SIZE = 256
CHANNELS = 1
BATCH_SIZE = 1
BUFFER_SIZE = 10

# Paths for TFRecords (update these paths as needed)
TRAIN_TFRECORD = 'data/train_images.tfrecords'
VAL_TFRECORD = 'data/val_images.tfrecords'

# ----- Data Loading and Preprocessing -----
def _parse_image_function(example_proto):
    image_feature_description = {
        'height':    tf.io.FixedLenFeature([], tf.int64),
        'width':     tf.io.FixedLenFeature([], tf.int64),
        'depth':     tf.io.FixedLenFeature([], tf.int64),
        'name' :     tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)

@tf.function
def read_and_decode(example):
    # Decode image bytes and reshape to [256,256,1]
    image_raw = tf.io.decode_raw(example['image_raw'], tf.int64)
    image_raw.set_shape([IMG_SIZE * IMG_SIZE])
    image = tf.reshape(image_raw, [IMG_SIZE, IMG_SIZE, CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 1024)
    
    # Decode label bytes and reshape to [256,256,1]
    label_raw = tf.io.decode_raw(example['label_raw'], tf.uint8)
    label_raw.set_shape([IMG_SIZE * IMG_SIZE])
    label = tf.reshape(label_raw, [IMG_SIZE, IMG_SIZE, 1])
    return image, label

def load_datasets():
    # Load TFRecord datasets
    raw_train_ds = tf.data.TFRecordDataset(TRAIN_TFRECORD)
    raw_val_ds   = tf.data.TFRecordDataset(VAL_TFRECORD)
    
    # Parse the TFRecords
    parsed_train_ds = raw_train_ds.map(_parse_image_function)
    parsed_val_ds   = raw_val_ds.map(_parse_image_function)
    
    # Decode images and labels
    train_ds = parsed_train_ds.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds   = parsed_val_ds.map(read_and_decode)
    
    # Set up pipeline: cache, shuffle, batch, and prefetch.
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = val_ds.batch(BATCH_SIZE)
    
    return train_ds, test_ds, parsed_train_ds, parsed_val_ds

# ----- Visualization Functions -----
def display(display_list, titles=['Input Image', 'Label', 'Prediction']):
    plt.figure(figsize=(10, 10))
    for i in range(len(display_list)):
        # Reshape to 256x256 for display
        img = tf.reshape(display_list[i], [IMG_SIZE, IMG_SIZE])
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    # Take argmax across channels to produce a mask (0 or 1)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset, num=1):
    for image, label in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], label[0], create_mask(pred_mask)])

# Custom Callback to display predictions after each epoch
class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        print("\nEpoch {} ended; sample predictions:".format(epoch + 1))
        show_predictions(self.model, self.test_dataset, num=1)

# ----- Model Building -----
def build_task1_model():
    # Task 1: Fully Connected NN with one hidden layer
    model = Sequential([
        Flatten(input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS]),
        Dense(64, activation='relu'),
        Dense(IMG_SIZE * IMG_SIZE * 2, activation='softmax'),
        Reshape((IMG_SIZE, IMG_SIZE, 2))
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def build_task2_model():
    # Task 2: Convolutional Neural Network
    model = Sequential([
        Conv2D(input_shape=[IMG_SIZE, IMG_SIZE, CHANNELS],
               filters=100, kernel_size=5, strides=2, padding="same",
               activation=tf.nn.relu, name="Conv1"),
        MaxPool2D(pool_size=2, strides=2, padding="same"),
        Conv2D(filters=200, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu),
        MaxPool2D(pool_size=2, strides=2, padding="same"),
        Conv2D(filters=300, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu),
        Conv2D(filters=300, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu),
        Conv2D(filters=2, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu),
        Conv2DTranspose(filters=2, kernel_size=31, strides=16, padding="same")
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def dice_coef(y_true, y_pred, smooth=1):
    # Compute Dice coefficient as a similarity metric
    indices = K.argmax(y_pred, axis=3)
    indices = K.reshape(indices, [-1, IMG_SIZE, IMG_SIZE, 1])
    indices_cast = K.cast(indices, dtype='float32')
    axis = [1, 2, 3]
    intersection = K.sum(y_true * indices_cast, axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(indices_cast, axis=axis)
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def build_task3_model():
    # Task 3: CNN with Dice metric as an evaluation metric
    model = build_task2_model()  # Use same architecture as Task 2
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[dice_coef, 'accuracy'])
    return model

# ----- Training Function -----
def train_model(model, train_ds, test_ds, epochs, steps_per_epoch, callbacks_list):
    history = model.fit(train_ds,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=test_ds,
                        callbacks=callbacks_list)
    return history

# ----- Main Script -----
def main(task):
    # Load data
    train_ds, test_ds, parsed_train_ds, parsed_val_ds = load_datasets()
    steps_per_epoch = len(list(parsed_train_ds))
    
    # Select model based on task argument
    if task == 1:
        print("Training Task 1 Model: Fully Connected Network")
        model = build_task1_model()
    elif task == 2:
        print("Training Task 2 Model: CNN")
        model = build_task2_model()
    elif task == 3:
        print("Training Task 3 Model: CNN with Dice Metric")
        model = build_task3_model()
    else:
        raise ValueError("Task must be 1, 2, or 3.")
    
    model.summary()
    
    # Set up callbacks: TensorBoard and DisplayCallback.
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    display_cb = DisplayCallback(test_ds)
    
    epochs = 20 if task != 3 else 30  # For Task 3, use 30 epochs; otherwise 20 epochs.
    history = train_model(model, train_ds, test_ds, epochs, steps_per_epoch, [display_cb, tensorboard_cb])
    
    # Plot training metrics.
    plt.figure()
    plt.plot(range(epochs), history.history['loss'], 'r', label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(range(epochs), history.history['val_loss'], 'bo', label='Validation Loss')
    if task == 3 and 'dice_coef' in history.history:
        plt.plot(range(epochs), history.history['dice_coef'], 'go', label='Dice Coefficient')
    plt.title('Training Metrics for Task {}'.format(task))
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    # Evaluate and display sample predictions.
    results = model.evaluate(test_ds)
    print("Evaluation results:", results)
    show_predictions(model, test_ds, num=5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Updated Image Segmentation Script")
    parser.add_argument('--task', type=int, default=1, 
                        help='Task number: 1 (Fully Connected), 2 (CNN), 3 (CNN with Dice Metric)')
    args = parser.parse_args()
    main(args.task)
