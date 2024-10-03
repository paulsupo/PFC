import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GRU, Bidirectional, BatchNormalization, Add
from tensorflow.keras.models import Model

vocab_size = 32  # Tamaño del vocabulario (puede cambiar según el dataset)
embedding_dim = 256  # Dimensión del embedding de caracteres
num_cbhg_kernels = 16  # Número de kernels para el CBHG
cbhg_gru_units = 128  # Número de celdas GRU en el CBHG
pre_net_units = [256, 128]  # Unidades en la red pre-net
decoder_gru_units = 256  # Unidades en las celdas GRU del decoder
post_cbhg_kernels = 8  # Número de kernels en el post-procesamiento
output_dim = 80  # Dimensión del espectrograma Mel de salida

# Módulo CBHG
# Modificación de cbhg_module para asegurar compatibilidad de dimensiones
def cbhg_module(inputs, num_kernels, num_highway_layers, gru_units):
    # Banco de convoluciones 1D
    conv_outputs = []
    for k in range(1, num_kernels + 1):
        conv = Conv1D(filters=128, kernel_size=k, activation='relu', padding='same')(inputs)
        conv_outputs.append(conv)
    conv_outputs = tf.keras.layers.concatenate(conv_outputs)
    
    # Max pooling y proyecciones
    max_pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')(conv_outputs)
    proj_1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(max_pool)
    proj_2 = Conv1D(filters=256, kernel_size=3, padding='same')(proj_1)  # Cambia el número de filtros para que coincida con inputs
    
    # Asegurar que proj_2 tenga las mismas dimensiones que inputs
    if proj_2.shape[-1] != inputs.shape[-1]:
        proj_2 = Dense(inputs.shape[-1])(proj_2)  # Cambiar la proyección para que coincida con inputs
    
    # Conexión residual
    residual = Add()([inputs, proj_2])
    
    # Capas Highway
    for _ in range(num_highway_layers):
        residual = Dense(units=128, activation='relu')(residual)
    
    # GRU bidireccional
    outputs = Bidirectional(GRU(units=gru_units, return_sequences=True))(residual)
    return outputs


# Encoder
def encoder(inputs):
    embedded = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    prenet = Dropout(0.5)(Dense(units=pre_net_units[0], activation='relu')(embedded))
    prenet = Dropout(0.5)(Dense(units=pre_net_units[1], activation='relu')(prenet))
    encoder_outputs = cbhg_module(prenet, num_cbhg_kernels, 4, cbhg_gru_units)
    return encoder_outputs

# Decoder
def decoder(encoder_outputs):
    prenet = Dropout(0.5)(Dense(units=pre_net_units[0], activation='relu')(encoder_outputs))
    prenet = Dropout(0.5)(Dense(units=pre_net_units[1], activation='relu')(prenet))
    gru_1 = GRU(units=decoder_gru_units, return_sequences=True)(prenet)
    gru_2 = Add()([gru_1, GRU(units=decoder_gru_units, return_sequences=True)(gru_1)])
    return gru_2

# Post-procesamiento para la síntesis de audio
def post_processing_net(inputs):
    post_cbhg_outputs = cbhg_module(inputs, post_cbhg_kernels, 4, cbhg_gru_units)
    linear_outputs = Dense(units=output_dim)(post_cbhg_outputs)
    return linear_outputs

# Definición del modelo completo de Tacotron
inputs = Input(shape=(None,), name='input_text')
encoder_outputs = encoder(inputs)
decoder_outputs = decoder(encoder_outputs)
post_outputs = post_processing_net(decoder_outputs)

# Construcción del modelo
tacotron_model = Model(inputs=inputs, outputs=post_outputs, name='Tacotron')
tacotron_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss='mean_squared_error')

# Guardar el modelo en un archivo para ser reutilizado
tacotron_model.save('tacotron_model.h5')

# Mostrar resumen del modelo
tacotron_model.summary()
