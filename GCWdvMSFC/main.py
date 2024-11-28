#train/test
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import math
from GCWdvMSFC import GCWdvMSFC
# Load data for training and validation
train_input_tc = np.load('./tr_longterm_demo.npy')  # Long-term input data for training
train_input_td = np.load('./tr_shortterm_demo.npy')  # Short-term input data for training
train_y = np.load('./tr_label_demo.npy')  # Training labels

val_input_tc = np.load('./val_longterm_demo.npy')  # Long-term input data for validation
val_input_td = np.load('./val_shortterm_demo.npy')  # Short-term input data for validation
val_y = np.load('./val_label_demo.npy')  # Validation labels

# Convert labels to categorical
train_y = tf.keras.utils.to_categorical(train_y, num_classes=18)
val_y = tf.keras.utils.to_categorical(val_y, num_classes=18)

# Hyperparameters
label_smoothing = 0.3
initial_learning_rate = 0.01
batch_size = 128
epochs = 20

# Define Learning Rate Scheduler
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

# Create GCWdxNet model (Assume the function `gcwdxnet_module` is already defined)
input_tc_shape = (train_input_tc.shape[1], train_input_tc.shape[2])
input_td_shape = (train_input_td.shape[1], train_input_td.shape[2])
model = GCWdxnet_module(input_tc_shape, input_td_shape)

# Compile the model
loss1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.AUTO)
adam = Adam(learning_rate=0.0)
model.compile(optimizer=adam, loss=loss1, metrics=['precision', tf.keras.metrics.AUC(name='AUC', multi_label=True)])

# Set up learning rate scheduler
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# Train the model
# Custom log loss calculation and confidence score function
def log_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

def confidence_score(y_pred):
    return tf.reduce_mean(tf.reduce_max(y_pred, axis=-1))
history = model.fit([train_input_tc, train_input_td], train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list,
                    validation_data=([val_input_tc, val_input_td], val_y), verbose=1, shuffle=True, class_weight={0: 0.20, 1: 1.})

# Evaluate model performance on validation set
val_predictions = model.predict([val_input_tc, val_input_td])
log_loss_value = log_loss(val_y, val_predictions).numpy()
confidence = confidence_score(val_predictions).numpy()

print(f'Log Loss: {log_loss_value}')
print(f'Confidence Score: {confidence}')

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# #model training
# # Model training with GPU support
# Uncomment the following code block to train the model using a GPU
# with tf.device("/gpu:0"):
#     loss1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.AUTO)
#     model = GCWdxnet_module(input_tc_shape, input_td_shape)
#     model.summary()
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000)
#     adam = Adam(learning_rate=lr_schedule)
#     model.compile(optimizer=adam, loss=loss1, metrics=['precision', 'recall', tf.keras.metrics.AUC(name='AUC', multi_label=True)])
#     # Alternatively, use SGD optimizer
#     # model.compile(optimizer=sgd, loss=loss1, metrics=['precision', 'recall', tf.keras.metrics.AUC(name='AUC', multi_label=True)])
#     # Or use focal loss with custom gamma and alpha
#     # model.compile(optimizer=adam, loss=[focal_loss(gamma=5, alpha=0.75)], metrics=['precision', 'recall', 'accuracy'])
#     history = model.fit([train_input_tc, train_input_td], train_y, batch_size=batch_size, epochs=epochs,
#                         validation_data=([val_input_tc, val_input_td], val_y), verbose=1, shuffle=True, class_weight={0: 0.20, 1: 1.})
#