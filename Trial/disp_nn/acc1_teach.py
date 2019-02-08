from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dot, Flatten, Input, Concatenate
import numpy
import data
import matplotlib.pyplot as plt
# constants
training_size = 288000
conv_feature_maps = 112
# fix random seed for reproducibility
numpy.random.seed(7)

# load Cones dataset
left, right, outputs = data.get_batch("../samples/cones/", 11, 3, 6, 4)
left = left[0:training_size]
right = right[0:training_size]
outputs = outputs[0:training_size]
# create model
left_input = Input(shape=(11, 11, 1, ))
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_input)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (left_conv)
left_flatten = Flatten()(left_conv)

right_input = Input(shape=(11, 11, 1, ))
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_input)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu") (right_conv)
right_conv = Conv2D(conv_feature_maps, kernel_size=3, activation="relu")(right_conv)
right_flatten = Flatten()(right_conv)

conc_layer = Concatenate()([left_flatten, right_flatten])
dense_layer = Dense(384, activation="relu")(conc_layer)
dense_layer = Dense(384, activation="relu")(dense_layer)
output_layer = Dense(1, activation="sigmoid")(dense_layer)
#output_layer = Dot(axes=-1, normalize = True)([left_flatten, right_flatten])

model = Model(inputs=[left_input, right_input], outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit([left,right], outputs, epochs=10, batch_size=128)
predictions = model.predict([left[0:10], right[0:10]])
print(predictions)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plot_model(model, show_shapes=True, to_file='model.png')
model.save_weights("weights/acc1_weights.h5")
########################################################################################

