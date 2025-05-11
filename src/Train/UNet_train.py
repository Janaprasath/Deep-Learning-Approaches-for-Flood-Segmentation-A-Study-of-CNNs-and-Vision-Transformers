

input_shape = (256, 256, 3)
model = unet_model(input_shape[0], input_shape[1], input_shape[2])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_loss,
              metrics=["accuracy",dice_coefficient, f1_score, mean_iou, precision, recall]

model_history=model.fit(train_generator, validation_data=val_generator, epochs=100)