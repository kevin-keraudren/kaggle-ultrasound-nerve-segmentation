from model import *

X,Y = get_training_data_classification("cropped_positives")
input_shape = X[0].shape
print("input_shape:", input_shape)
model = get_model_classification(input_shape=input_shape,
                                 filename="callback_model/model_epoch00110_weights.h5",
                                 start_epoch=101)

print(X.shape,Y.shape)

model.fit(np.array(X),np.array(Y),
          nb_epoch=1000, 
          batch_size=10,
          callbacks=[
              MyCallbackClassification(),
              TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True)
          ])

save_model(model, "saved_model_classification/model")