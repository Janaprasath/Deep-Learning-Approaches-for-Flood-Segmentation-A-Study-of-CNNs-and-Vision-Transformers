
config = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "num_patches": (256 // 16) ** 2,
    "hidden_dim": 768,   n
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "dropout_rate": 0.1,
}

model = build_unetr_2d(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_loss,
              metrics=["accuracy",dice_coefficient, f1_score, mean_iou, precision, recall])

model_history=model.fit(train_generator, validation_data=val_generator, epochs=20)