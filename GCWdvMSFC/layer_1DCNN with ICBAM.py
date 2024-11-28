# 1D CNN with ResBlock-CBAM Model for Short-Term Period Feature Extraction
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, GlobalAveragePooling1D, Dense, Reshape, Activation
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam

# 1. CBAM module
# Convolutional Block Attention Module (CBAM) including both channel and spatial attention
def cbam_block(input_feature, ratio=8):
    # Channel Attention Module
    channel = input_feature.shape[-1]
    avg_pool = GlobalAveragePooling1D()(input_feature)
    max_pool = GlobalMaxPooling1D()(input_feature)
    shared_dense = Dense(channel // ratio, activation='relu')
    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)
    channel_out = Dense(channel, activation='sigmoid')(avg_out + max_out)
    channel_out = Reshape((1, channel))(channel_out)
    channel_refined = Multiply()([input_feature, channel_out])

    # Spatial Attention Module
    avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    spatial_out = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_out = Conv1D(1, kernel_size=7, activation='sigmoid', padding='same')(spatial_out)
    spatial_refined = Multiply()([channel_refined, spatial_out])

    return spatial_refined


# 2. Residual Block with CBAM
def resblock_cbam(input_feature, filters, kernel_size=3):
    conv_out = Conv1D(filters, kernel_size=kernel_size, padding='same', activation='relu')(input_feature)
    conv_out = Conv1D(filters, kernel_size=kernel_size, padding='same', activation='relu')(conv_out)
    cbam_out = cbam_block(conv_out)
    res_out = Add()([input_feature, cbam_out])  # Residual connection
    return res_out


# 3. Build the model
def build_model(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape)

    # 4. Initial convolution block
    conv_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)

    # 5. Residual Block with CBAM
    res_cbam_out = resblock_cbam(conv_out, filters=64)

    # 6. Next convolution block
    next_conv_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(res_cbam_out)

    # 7. Global pooling to extract fused features
    global_avg_pooling = GlobalAveragePooling1D()(next_conv_out)

    # Build the model
    model = Model(inputs=input_layer, outputs=global_avg_pooling)

    return model


# 8. Build the model
# The data shape is (5000, 100, 4), so the input shape is (100, 4)
input_shape = (data.shape[1], data.shape[2])
model = build_model(input_shape)

# 9. Model summary
# Print model structure information
model.summary()

# 10. Extract fused features
# Assume input data X_data is ready (e.g., the loaded data)
fused_features = model.predict(data)
print("Fused Features (Td):")
print(fused_features)



