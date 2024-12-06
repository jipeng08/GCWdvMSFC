#AAW features coupling and wear diagnosis
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, Multiply, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam


# 1. Define AAW module (Adaptive Attention Weighting)
def aaw_module(tc, td):
    # Multi-Head Attention to calculate feature importance
    attention_tc = MultiHeadAttention(num_heads=4, key_dim=64)(tc, tc)
    attention_td = MultiHeadAttention(num_heads=4, key_dim=64)(td, td)

    # Adaptive Attention Weighting (AAW) to fuse features
    tc_weight = Dense(1, activation='sigmoid')(attention_tc)
    td_weight = Dense(1, activation='sigmoid')(attention_td)

    # Apply weights to respective features
    weighted_tc = Multiply()([tc, tc_weight])
    weighted_td = Multiply()([td, td_weight])

    # Concatenate weighted features to produce final fused feature
    fused_feature = Concatenate(axis=-1)([weighted_tc, weighted_td])
    return fused_feature


# 2. Define model inputs
# Inputs come directly from Tc and Td modules from the previous models
input_tc_shape = (None, 64)
input_td_shape = (None, 64)

input_tc = Input(shape=input_tc_shape)
input_td = Input(shape=input_td_shape)

# 3. Fuse Tc and Td using AAW
fused_feature = aaw_module(input_tc, input_td)

# 4. Global pooling and final output
pooled_feature = GlobalAveragePooling1D()(fused_feature)
final_output = Dense(18, activation='softmax')(pooled_feature)  # Output for three classes of wear

# 5. Build and compile the model
model = Model(inputs=[input_tc, input_td], outputs=final_output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Model summary
model.summary()

# 7. Train the model
#  Tc and Td are provided from previously computed outputs
# For demonstration purposes, dummy data is used here to represent Tc and Td
x_train_tc = np.random.rand(y_train.shape[0], 100, 128)  # Example Tc output (combine data for 3 types of labels)
x_train_td = np.random.rand(y_train.shape[0], 100, 64)  # Example Td output (combine data for 3 types of labels)

# Load labels for the three types of cathode wear
labels_no_wear = np.load('/GCWdvMSFC/tr_no wear_initial_demo.npy')
labels_steel_bar_corrosion = np.load('/GCWdvMSFC/tr_steel bar_corrosion_demo.npy')
labels_surface_wear = np.load('/GCWdvMSFC/tr_surface wear_demo.npy')

# Combine labels into one array (assuming categorical labels)
y_train = np.concatenate([labels_no_wear, labels_steel_bar_corrosion, labels_surface_wear], axis=0)

# One-hot encode labels (assuming 3 classes)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=18)

# Train the model
history = model.fit([x_train_tc, x_train_td], y_train, epochs=20, batch_size=64, validation_split=0.2)

# 8. Predict the cathode wear label based on fused features
predictions = model.predict([x_train_tc, x_train_td])
predicted_labels = np.argmax(predictions, axis=1)
print(f'Predicted Labels:\n{predicted_labels}')

