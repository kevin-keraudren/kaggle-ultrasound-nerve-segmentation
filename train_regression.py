from model import *

X,Y = get_training_data_regression("cropped_positives")
input_shape = X[0].shape
print("input_shape:", input_shape)
model = get_model_regression(input_shape=input_shape)

print(X.shape,Y.shape)

model.fit(np.array(X),np.array(Y),
          nb_epoch=1000, 
          batch_size=10,
          callbacks=[
              MyCallbackRegression(),
              TensorBoard(log_dir='./logs_regression', histogram_freq=0, write_graph=True)
          ])

save_model(model, "saved_model_regression/model")