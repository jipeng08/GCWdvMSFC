# BiLSTM-CNN Long-trem Trend Feature Tc Extraction Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Attention, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, LayerNormalization
from tensorflow.keras.optimizers import Adam



# 1. Build the model
def build_model(input_shape):
    # Input layer
    input_layer = Input(shape=input_shape)

    # 2. BiLSTM layer to extract long-term trend features (Tc)
    # Use Bidirectional LSTM to extract long-term features from time series data
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(input_layer)  # Output shape: (None, timesteps, 128)


    # Reshape BiLSTM output to match CNN input shape
    reshaped_lstm_out = Reshape(input_shape)(lstm_out)  # Adjust to match CNN input: (None, timesteps, features)

    # Example usage
    input_shape = (100, 32)  # Example: 100 timesteps, 32 features
    model = build_model(input_shape)
    model.summary()

    # 3. CNN layer to extract local features (Ts)
    # Use two 1D convolutional layers to extract local patterns with ReLU activation function
    cnn_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(
        input_layer)  # Output shape: (None, timesteps, 64)
    cnn_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(
        cnn_out)  # Output shape: (None, timesteps, 64)

    # 4. Add 1x1 convolution (SCconv) module
    # SCconv module for feature compression to enhance model representation and reduce computational cost
    scconv_out = Conv1D(64, kernel_size=1, activation='relu', padding='same')(cnn_out)

    # 5. Merge LSTM and CNN features
    # Use Add layer to merge lstm_out and scconv_out features
    merged = Add()([lstm_out, scconv_out])

    # 6. Multi-head attention module to fuse features
    # Use multi-head attention to process the merged features and capture important relationships between time steps
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(merged, merged)
    attention_output = LayerNormalization()(attention_output)  # Layer normalization for stability

    # 7. Output the fused features (Tc)
    # Use global average pooling to extract the final fused features
    global_avg_pooling = GlobalAveragePooling1D()(attention_output)

    # Build the model
    model = Model(inputs=input_layer, outputs=global_avg_pooling)

    return model


# 8. Build the model
input_shape = (data.shape[1], data.shape[2])
model = build_model(input_shape)

# 9. Model summary
# Print model structure information
model.summary()

# 10. Extract fused features (Tc)
# Assume the input data is ready
fused_features = model.predict(data)
print(f'Fused Features (Tc):\n{fused_features}')


