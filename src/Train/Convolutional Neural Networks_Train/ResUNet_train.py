

input_shape = (256, 256, 3)
model = build_resunet(input_shape)
model.compile(optimizer="adam",
              loss=combined_loss,
              metrics=["accuracy", dice_coefficient, f1_score, mean_iou, precision, recall])

model_history=model.fit(train_generator, validation_data=val_generator, epochs=100)