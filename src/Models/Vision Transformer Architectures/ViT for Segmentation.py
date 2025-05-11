class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=cf["num_heads"], key_dim=cf["hidden_dim"])(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Add()([x, skip_2])

    return x

def extract_patches(inputs, patch_size):
    return tf.image.extract_patches(
        inputs, sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding="VALID"
    )

def ViT_Segmentation(cf):

    input_shape = (cf["image_size"], cf["image_size"], cf["num_channels"])
    inputs = Input(input_shape)

    num_patches = (cf["image_size"] // cf["patch_size"]) ** 2
    flattened_patch_dim = cf["patch_size"] * cf["patch_size"] * cf["num_channels"]

    patches = Lambda(lambda x: extract_patches(x, cf["patch_size"]))(inputs)
    patches = Reshape((num_patches, flattened_patch_dim))(patches)
    patch_embed = Dense(cf["hidden_dim"])(patches)  # Linear Projection

    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = Embedding(input_dim=num_patches, output_dim=cf["hidden_dim"])(positions)
    embed = patch_embed + pos_embed

    x = embed
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    x = Reshape((cf["image_size"] // cf["patch_size"], cf["image_size"] // cf["patch_size"], cf["hidden_dim"]))(x)

    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    outputs = Conv2DTranspose(cf["num_classes"], (1, 1), activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model