import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, Attention, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten, LayerNormalization
from tensorflow.keras.layers import Input, Conv1D, Multiply, Add, GlobalAveragePooling1D, Dense, Reshape, Activation
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, Multiply, Concatenate, GlobalAveragePooling1D
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pykeen
from pykeen.pipeline import pipeline
from pykeen.models import TransR
from pykeen.triples import TriplesFactory
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, MultiHeadAttention, Add, Multiply, Concatenate, GlobalAveragePooling1D, Dense, LayerNormalization, Flatten, TimeDistributed
from tensorflow.keras.layers import Embedding
import tensorflow_addons as tfa
import os


# BiLSTM-CNN Long-trem Trend Feature Tc Extraction Model
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

# 1D CNN with ResBlock-CBAM Model for Short-Term Period Feature Extraction

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

# CRF with TransR layer knowledgement Extraction Model

# Step 1: Prepare the aluminum electrolyzer metrics data
data = {
    'average_voltage': [4.0, 4.1, 4.2, 4.0, 4.3],
    'electrolyte_level': [10.0, 9.8, 10.2, 9.9, 10.1],
    'fluoride_salt': [2.5, 2.6, 2.4, 2.5, 2.6],
    'anode_travel': [50.0, 55.0, 52.0, 53.0, 51.0],
    'aluminum_output': [100, 110, 105, 98, 103],
    'power_consumption': [500, 510, 520, 490, 515]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Step 2: Build the Knowledge Graph using RDFLib
g = Graph()
namespace = URIRef("http://example.org/")

for i, row in df.iterrows():
    # Create entities for each metric and value pair
    voltage = URIRef(f"{namespace}average_voltage_{i}")
    level = URIRef(f"{namespace}electrolyte_level_{i}")
    salt = URIRef(f"{namespace}fluoride_salt_{i}")
    anode = URIRef(f"{namespace}anode_travel_{i}")
    output = URIRef(f"{namespace}aluminum_output_{i}")
    power = URIRef(f"{namespace}power_consumption_{i}")

    # Add entities and their values to the graph
    g.add((voltage, RDF.type, RDFS.Class))
    g.add((voltage, URIRef(f"{namespace}has_value"), Literal(row['average_voltage'])))

    g.add((level, RDF.type, RDFS.Class))
    g.add((level, URIRef(f"{namespace}has_value"), Literal(row['electrolyte_level'])))

    g.add((salt, RDF.type, RDFS.Class))
    g.add((salt, URIRef(f"{namespace}has_value"), Literal(row['fluoride_salt'])))

    g.add((anode, RDF.type, RDFS.Class))
    g.add((anode, URIRef(f"{namespace}has_value"), Literal(row['anode_travel'])))

    g.add((output, RDF.type, RDFS.Class))
    g.add((output, URIRef(f"{namespace}has_value"), Literal(row['aluminum_output'])))

    g.add((power, RDF.type, RDFS.Class))
    g.add((power, URIRef(f"{namespace}has_value"), Literal(row['power_consumption'])))

# Print triples in the graph
print("Knowledge Graph Triples:")
for s, p, o in g:
    print(s, p, o)

# Step 3: Use CRF for Relationship Modeling
# Prepare training data for CRF
X_train = [[(key, row[key]) for key in row.index] for i, row in df.iterrows()]
y_train = [["has_value"] * len(row) for row in df.iterrows()]  # Simple label for relationships

# Define and train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, c2=0.1,  # Regularization parameters
    max_iterations=100
)
crf.fit(X_train, y_train)

# Predict the relationships for the training data
y_pred = crf.predict(X_train)

# Evaluate the CRF model
print("\nCRF Model Evaluation:")
print(metrics.flat_classification_report(y_train, y_pred))

# Step 4: Knowledge Graph Embedding using TransR (PyKEEN)
# Create triples from the knowledge graph for embedding
triples = [
    ('average_voltage_1', 'has_value', 4.0),
    ('electrolyte_level_1', 'has_value', 10.0),
    ('fluoride_salt_1', 'has_value', 2.5),
    ('anode_travel_1', 'has_value', 50.0),
    ('aluminum_output_1', 'has_value', 100),
    ('power_consumption_1', 'has_value', 500),
    # Add additional triples for other rows as needed
]

# Create a TriplesFactory object
triples_factory = TriplesFactory.from_labeled_triples(triples)

# Run the TransR model to learn embeddings
result = pipeline(
    model=TransR,
    dataset=triples_factory,
    training_loop='sls',  # Use Stochastic Local Search for optimization
    num_epochs=100,
)

# Get the embeddings for entities and relations
embeddings = result.model.entity_embeddings.weight.data
print("\nTransR Entity Embeddings:")
print(embeddings)

#AAW features coupling and wear diagnosis

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
input_tc_shape = (None, 128)  # Assuming output shape from the Tc module is (timesteps, 128)
input_td_shape = (None, 64)  # Assuming output shape from the Td module is (timesteps, 64)

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

#train/test

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
import math

# Load data for training and validation
train_input_tc = np.load('/GCWdvMSFC/tr_long-term trend_demo.npy')  # Long-term input data for training
train_input_td = np.load('/GCWdvMSFC/tr_short-term period_demo.npy')  # Short-term input data for training
train_y = np.load('/GCWdvMSFC/tr_label_demo.npy')  # Training labels

val_input_tc = np.load('/GCWdvMSFC/val_longterm_demo.npy')  # Long-term input data for validation
val_input_td = np.load('/GCWdvMSFC/val_shortterm_demo.npy')  # Short-term input data for validation
val_y = np.load('/GCWdvMSFC/val_label_demo.npy')  # Validation labels

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
model = gcwdxnet_module(input_tc_shape, input_td_shape)  # You need to define gcwdxnet_module in the same script or import it
model.summary()

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




