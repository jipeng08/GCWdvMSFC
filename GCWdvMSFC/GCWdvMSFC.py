#GCWdvMSFC
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, MultiHeadAttention, Add, Multiply, Concatenate, GlobalAveragePooling1D, Dense, LayerNormalization, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding
import tensorflow_addons as tfa

# 1. Define BiLSTM-CNN module for long-term trend features (Tc)
def bilstm_cnn_module(input_shape):
    input_layer = Input(shape=input_shape)
    # BiLSTM for long-term trend extraction
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    # CNN for local feature extraction
    cnn_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    cnn_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(cnn_out)
    # 1x1 convolution (SCconv) for feature compression
    scconv_out = Conv1D(64, kernel_size=1, activation='relu', padding='same')(cnn_out)
    # Merge LSTM and CNN features
    merged = Add()([lstm_out, scconv_out])
    # Multi-Head Attention for feature refinement
    attention_out = MultiHeadAttention(num_heads=4, key_dim=64)(merged, merged)
    attention_out = LayerNormalization()(attention_out)
    return Model(inputs=input_layer, outputs=attention_out, name='BiLSTM_CNN_Module')

# 2. Define ResBlock-CBAM module for short-term period features (Td)
def resblock_cbam_module(input_shape):
    input_layer = Input(shape=input_shape)
    # Initial convolution block
    conv_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
    # Residual Block with CBAM
    res_out = Conv1D(64, kernel_size=3, padding='same', activation='relu')(conv_out)
    res_out = Conv1D(64, kernel_size=3, padding='same', activation='relu')(res_out)
    # Channel Attention Module
    channel = res_out.shape[-1]
    avg_pool = GlobalAveragePooling1D()(res_out)
    max_pool = GlobalAveragePooling1D()(res_out)
    shared_dense = Dense(channel // 8, activation='relu')
    avg_out = shared_dense(avg_pool)
    max_out = shared_dense(max_pool)
    channel_out = Dense(channel, activation='sigmoid')(avg_out + max_out)
    channel_out = tf.reshape(channel_out, (-1, 1, channel))
    channel_refined = Multiply()([res_out, channel_out])
    # Spatial Attention Module
    avg_pool_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    spatial_out = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    spatial_out = Conv1D(1, kernel_size=7, activation='sigmoid', padding='same')(spatial_out)
    spatial_refined = Multiply()([channel_refined, spatial_out])
    # Residual connection
    res_cbam_out = Add()([input_layer, spatial_refined])
    return Model(inputs=input_layer, outputs=res_cbam_out, name='ResBlock_CBAM_Module')

# 3. Define AAW module (Adaptive Attention Weighting)
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

# 4. Define CRF and TransR module for knowledge-based feature extraction
def crf_transr_module(input_shape):
    input_layer = Input(shape=input_shape)
    # Embedding layer for knowledge representation
    embedding_out = Embedding(input_dim=1000, output_dim=64)(input_layer)
    # CRF Layer
    crf_out = tfa.layers.CRF(64)(embedding_out)
    # TransR component (assuming it's a dense layer for relational learning)
    transr_out = Dense(64, activation='relu')(crf_out)
    return Model(inputs=input_layer, outputs=transr_out, name='CRF_TransR_Module')

# 5. Define GCWdxNet module (combining BiLSTM-CNN, ResBlock-CBAM, AAW, and CRF-TransR)
def gcwdxnet_module(input_tc_shape, input_td_shape, input_knowledge_shape):
    # Tc (long-term features) from BiLSTM-CNN module
    bilstm_cnn_model = bilstm_cnn_module(input_tc_shape)
    input_tc = Input(shape=input_tc_shape)
    tc_output = bilstm_cnn_model(input_tc)

    # Td (short-term features) from ResBlock-CBAM module
    resblock_cbam_model = resblock_cbam_module(input_td_shape)
    input_td = Input(shape=input_td_shape)
    td_output = resblock_cbam_model(input_td)

    # Knowledge-based features from CRF-TransR module
    crf_transr_model = crf_transr_module(input_knowledge_shape)
    input_knowledge = Input(shape=input_knowledge_shape)
    knowledge_output = crf_transr_model(input_knowledge)

    # Fuse Tc and Td using AAW
    fused_feature = aaw_module(tc_output, td_output)

    # Calculate difference between AAW fused features and knowledge-based features
    knowledge_difference = tf.reduce_mean(tf.abs(fused_feature - knowledge_output), axis=-1)
    knowledge_mask = tf.where(knowledge_difference > 0.3, 1.0, 0.0)  # Mask where difference exceeds 30%

    # Apply knowledge-based features to assist AAW decision if difference is high
    adjusted_feature = Multiply()([knowledge_output, knowledge_mask])
    final_fused_feature = Add()([fused_feature, adjusted_feature])

    # Global pooling and final output
    pooled_feature = GlobalAveragePooling1D()(final_fused_feature)
    final_output = Dense(18, activation='softmax')(pooled_feature)  # Output for 18 classes

    # Build and compile the model
    model = Model(inputs=[input_tc, input_td, input_knowledge], outputs=final_output, name='GCWdxNet_Module')
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 6. Data Input for Real-Time Signal
# Assuming real-time signal input with shape (timesteps, features)
# For demonstration purposes, use None for batch size to allow dynamic input
input_tc_shape = (None, 128)  # Placeholder for Tc module input
input_td_shape = (None, 64)   # Placeholder for Td module input
input_knowledge_shape = (None,)  # Placeholder for knowledge input

# Create GCWdxNet model
gcwdxnet_model = gcwdxnet_module(input_tc_shape, input_td_shape, input_knowledge_shape)

# Model summary
gcwdxnet_model.summary()

# Example real-time data input (dummy data for demonstration)
x_real_time_tc = np.random.rand(1, 100, 128)  # Example input for Tc
x_real_time_td = np.random.rand(1, 100, 64)   # Example input for Td
x_real_time_knowledge = np.random.randint(0, 1000, (1, 100))  # Example input for knowledge features

# Predict using real-time signal input
predictions = gcwdxnet_model.predict([x_real_time_tc, x_real_time_td, x_real_time_knowledge])
print(f'Predicted Labels:\n{np.argmax(predictions, axis=1)}')

