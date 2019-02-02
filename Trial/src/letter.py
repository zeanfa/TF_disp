from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load letters dataset
dataset = numpy.loadtxt("../samples/letters/letters.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:49]
Y = dataset[:,49:52]

# create model
model = Sequential()
model.add(Dense(15, input_dim=49, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

# complile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model
#model.fit(X, Y, epochs=1000, batch_size=3)
# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
#predictions = model.predict(X[0:1])
predictions = model.predict(X[0:1])
print(predictions)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(rounded)
