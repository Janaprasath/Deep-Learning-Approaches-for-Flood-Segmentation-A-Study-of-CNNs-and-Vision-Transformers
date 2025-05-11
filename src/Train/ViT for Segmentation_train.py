
cf = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 6,
    "dropout_rate": 0.1,
    "num_classes": 1
}
model = ViT_Segmentation(cf)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_loss,
              metrics=["accuracy",dice_coefficient, f1_score, mean_iou, precision, recall])

model_history=model.fit(train_generator, validation_data=val_generator, epochs=100)