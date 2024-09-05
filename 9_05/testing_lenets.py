import numpy as np
from scipy.io import loadmat  
from tensorflow.keras.datasets import mnist,fashion_mnist,cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input,Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.optimizers import SGD, Adam
from scipy.signal import convolve2d
import argparse
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift,ifftshift
from keras_flops import get_flops
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import get_value


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('val_accuracy'))

# Initialize the accuracy history callback
accuracy_history = AccuracyHistory()

def average_last_epochs(accuracies, N=5):
    return np.mean(accuracies[-N:])*100

def make_lr_decay_function(epochs):
    def lr_time_based_decay(epoch, lr):
        decay = epoch / (epochs * 0.5)/3
        new_lr = lr / (1+decay)
        return new_lr
    return lr_time_based_decay

num_of_subpixels=4

filters = loadmat('filters.mat')['filters']  

parser = argparse.ArgumentParser(description='Load MNIST or Fashion MNIST dataset.')

parser.add_argument('--dataset', '-d',type=str, required=True, help='Dataset to load ("mnist" or "fashion_mnist")')

args = parser.parse_args()
if args.dataset=='cifar':
    pic_size=32
else:
    pic_size=28
def load_dataset(dataset_name):
    if dataset_name == 'dmnist':
        return mnist.load_data()
    elif dataset_name == 'fmnist':
        return fashion_mnist.load_data()
    elif dataset_name=='cifar':
        return cifar10.load_data()
    else:
        raise ValueError("Invalid dataset name. Enter dmnist or fmnist or cifar")
    
def convert_to_grayscale(data):
    (train_images, train_labels), (test_images, test_labels) = data
    # Convert to grayscale by averaging channels
    train_images = np.mean(train_images, axis=3, keepdims=True)
    test_images = np.mean(test_images, axis=3, keepdims=True)
    return (train_images, train_labels), (test_images, test_labels)


def apply_filters(data, filters, pic_size):
    filtered_data = np.zeros((data.shape[0], pic_size, pic_size, num_of_subpixels)) 
    for i in range(data.shape[0]):
        for j in range(num_of_subpixels):  
            filtered_image = convolve2d(data[i, :, :, 0], filters[:, :, j], mode='same', boundary='fill')
            filtered_data[i, :, :, j] = filtered_image
    return filtered_data

# def apply_filters_and_display(data, filters, pic_size):
#     filtered_data = np.zeros((data.shape[0], pic_size, pic_size, num_of_subpixels))
#     for i in range(data.shape[0]):
#         fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # Setting up a subplot for original and num_of_subpixels filters
#         axes[0].imshow(data[i, :, :, 0], cmap='gray')
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
#         for j in range(1):
#             filtered_image = fft_convolve2d(data[i, :, :, 0], filters[:, :, j], mode='same', boundary='symm')
#             filtered_data[i, :, :, j] = filtered_image
#             axes[j + 1].imshow(filtered_image, cmap='gray')  # Display each filtered image
#             axes[j + 1].set_title(f'Filter {j + 1}')
#             axes[j + 1].axis('off')
#             filtered_image = convolve2d(data[i, :, :, 0], filters[:, :, j], mode='same', boundary='symm')
#             filtered_data[i, :, :, j] = filtered_image
#             axes[j + 2].imshow(filtered_image, cmap='gray')  # Display each filtered image
#             axes[j + 2].set_title(f'Filter {j + 1}')
#             axes[j + 2].axis('off')
#         plt.show()
#     return filtered_data



# def fft_convolve2d(image, kernel, mode='same', boundary='symm'):
#     pad_height = kernel.shape[0] // 2
#     pad_width = kernel.shape[1] // 2
#     if boundary == 'symm':
#         padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'reflect')
#     elif boundary =='fill':
#         padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant',constant_values=(0))
#     else:
#         padded_image=image
        

#     # Pad kernel to the size of the padded image
#     padded_kernel = np.zeros_like(padded_image)
#     padded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
    
#     # Roll the kernel to center it
#     k_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
#     padded_kernel = np.roll(padded_kernel, (-k_center[0], -k_center[1]), axis=(0, 1))

#     # FFT of image and kernel
#     image_fft = fft2(padded_image)
#     kernel_fft = fft2(padded_kernel)

#     # Element-wise multiplication and inverse FFT
#     convolved = ifft2(image_fft * kernel_fft).real

#     # Crop to original image size if mode is 'same'
#     if mode == 'same':
#         convolved = convolved[pad_height-1:-(pad_height+1), pad_width-1:-(pad_width+1)]

#     return convolved


if args.dataset=='cifar':
    (trainData, trainLabels), (testData, testLabels) = convert_to_grayscale(load_dataset(args.dataset))

else:
    (trainData, trainLabels), (testData, testLabels) = load_dataset(args.dataset)


trainData = trainData.reshape((trainData.shape[0], pic_size, pic_size, 1))
testData = testData.reshape((testData.shape[0], pic_size, pic_size, 1))

# # Apply MATLAB filters
trainData = apply_filters(trainData, filters,pic_size)
testData = apply_filters(testData, filters,pic_size)

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

trainLabels = to_categorical(trainLabels, 10)
testLabels = to_categorical(testLabels, 10)

model = Sequential([
    
    Input(shape=(pic_size, pic_size,num_of_subpixels)),
    # Input(shape=(pic_size, pic_size,1)),
    # Conv2D(4,(9,9), padding='same'),
    Activation('relu'),
    BatchNormalization(),
	MaxPooling2D(pool_size=(2, 2)),
    # Conv2D(16,(5,5), padding='valid'),
    # Activation('relu'),
    # BatchNormalization(),
	# MaxPooling2D(pool_size=(2, 2)),
    # Conv2D(16,(5,5), padding='valid'),
    # Activation('relu'),
    # BatchNormalization(),
	# MaxPooling2D(pool_size=(2, 2)),
    Flatten(), 
    # Dense(200),
    # Activation('relu'),
    # BatchNormalization(),
    # Dense(120),
    # Activation('relu'),
    # BatchNormalization(),
    # Dense(50),
    # Activation('relu'),
    # BatchNormalization(),
    Dense(10),
    Activation('softmax')
])

epochs = 25
epochs_averaged_over = 5
lr_decay = make_lr_decay_function(epochs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
print("[INFO] training...")
model.fit(trainData, trainLabels, batch_size=128, epochs=epochs,validation_data=(testData, testLabels),
    verbose=1,callbacks=[accuracy_history,LearningRateScheduler(lr_decay)])

print("[INFO] evaluating...")
(loss, accuracy) = model.evaluate(testData, testLabels,
    batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

average_accuracy = average_last_epochs(accuracy_history.acc, N=epochs_averaged_over)
print(f"Average accuracy over the last", epochs_averaged_over, f"epochs: {average_accuracy:.2f}%")


# model.summary()
# flops = get_flops(model, batch_size=1)
# print(f"FLOPs: {flops}")




# initial_learning_rate = 0.001
# epochs = 25
# total_steps = len(trainData) // 128 * epochs  # total number of steps in all epochs
# decay_steps = total_steps / epochs  


# decay_rate = 0.001 ** (1 / epochs)  

# lr_schedule = ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=decay_steps,
#     decay_rate=decay_rate,
#     staircase=False)
# model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
# print("[INFO] training...")
# model.fit(trainData, trainLabels, batch_size=128, epochs=epochs, validation_data=(testData, testLabels),
#     verbose=1, callbacks=[accuracy_history])
# print("[INFO] evaluating...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
#     batch_size=128, verbose=1)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))



# epochs = 25
# initial_learning_rate = 0.001
# decay_steps = len(trainData) // 128 * epochs
# alpha = 0.001

# lr_schedule = CosineDecay(
#     initial_learning_rate=initial_learning_rate,
#     decay_steps=decay_steps,
#     alpha=alpha
# )
# class LearningRateLogger(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
    
#         lr_schedule = self.model.optimizer.learning_rate
#         current_step = epoch * len(trainData) // 128
       
#         lr = lr_schedule(current_step)
#         formatted_lr = "{:.2e}".format(lr.numpy())
#         print(f"Learning Rate: {formatted_lr}")

# model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
# print("[INFO] training...")
# model.fit(trainData, trainLabels, batch_size=128, epochs=epochs, validation_data=(testData, testLabels),
#     verbose=1, callbacks=[accuracy_history,LearningRateLogger()])
# print("[INFO] evaluating...")
# (loss, accuracy) = model.evaluate(testData, testLabels,
#     batch_size=128, verbose=1)
# print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


# epochs=25
# epochs_averaged_over = 5
# def step_decay(epoch):
#     initial_lr = 0.001
#     drop = 1/5.62 
#     epochs_drop = 5  
#     lr = initial_lr * (drop ** (epoch // epochs_drop))
#     return lr

# lr_schedule = LearningRateScheduler(step_decay)

# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(trainData, trainLabels, batch_size=128, epochs=epochs, validation_data=(testData, testLabels),
#           verbose=1, callbacks=[lr_schedule])
# average_accuracy = average_last_epochs(accuracy_history.acc, N=epochs_averaged_over)
# print(f"Average accuracy over the last", epochs_averaged_over, f"epochs: {average_accuracy:.2f}%")