def swin_transformer_block(x, num_heads, key_dim, window_size, channels):
    x_norm = layers.LayerNormalization()(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_norm, x_norm)
    attn_output = layers.Conv2D(channels, (1, 1), padding='same')(attn_output)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    mlp = tf.keras.Sequential([
        layers.Dense(4 * channels, activation='gelu'),
        layers.Dense(channels)
    ])(x)
    x = layers.Add()([x, mlp])
    return x

# Swin U-Net Encoder Block
def swin_unet_encoder(x, num_blocks, num_heads, key_dim, window_size, channels):
    x = layers.Conv2D(channels, (3, 3), padding='same', activation='relu')(x)
    for _ in range(num_blocks):
        x = swin_transformer_block(x, num_heads, key_dim, window_size, channels)
    return x


def swin_unet_decoder(x, skip_connection, num_blocks, num_heads, key_dim, window_size, channels):
    x = layers.UpSampling2D((2, 2))(x)
    skip_connection = layers.Conv2D(channels, (1, 1), padding='same')(skip_connection)
    x = layers.Concatenate()([x, skip_connection])
    x = swin_unet_encoder(x, num_blocks, num_heads, key_dim, window_size, channels)
    return x


def build_swin_unet(input_shape=(128,128, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)  # Reduced channels to 32
    skip1 = swin_unet_encoder(x, num_blocks=1, num_heads=2, key_dim=32, window_size=7, channels=32)
    x = layers.MaxPooling2D((2, 2))(skip1)

    skip2 = swin_unet_encoder(x, num_blocks=1, num_heads=4, key_dim=64, window_size=7, channels=64)
    x = layers.MaxPooling2D((2, 2))(skip2)

    bottleneck = swin_unet_encoder(x, num_blocks=2, num_heads=8, key_dim=128, window_size=7, channels=128)

    x = swin_unet_decoder(bottleneck, skip2, num_blocks=1, num_heads=4, key_dim=64, window_size=7, channels=64)
    x = swin_unet_decoder(x, skip1, num_blocks=1, num_heads=2, key_dim=32, window_size=7, channels=32)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model