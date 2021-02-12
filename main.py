# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor
import pandas as pd

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Connect your script to Neptune
if os.getenv('CI') == "true":
    neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'), project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))
    neptune.create_experiment(name='shape_prediction')

# load the dataset
#read in data using pandas
all_df = pd.read_csv('data.csv')
#check data has been read in properly
print(all_df.head())

#create a dataframe with all training data except the target columns
all_X = all_df.drop(columns=['ellipsoid', 'cylinder', 'sphere'])
#check that the target variable has been removed
print(all_X.head())

#create a dataframe with only the target column
all_y = all_df[['ellipsoid', 'cylinder', 'sphere']]
#view dataframe
print(all_y.head())

#get number of columns in training data
train_cols = all_X.shape[1]
#get number of columns in output data
out_cols = all_y.shape[1]

print(train_cols)
print(out_cols)
last = len(all_X)
print(last)

split = int(last * 0.75)

print(split)
train_X = all_X[:split]
train_y = all_y[:split]
test_X = all_X[split:]
test_y = all_y[split:]
print(len(train_X))
print(len(test_X))

#normalise training data
train_mean = train_X.mean()
train_std = train_X.std()
print(train_mean)
print(train_std)



train_X = (train_X - train_mean) / train_std
test_X = (test_X - train_mean) / train_std
print(train_X.head())
print(test_X.head())


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=train_cols, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)

#train model
if os.getenv('CI') == "true":
    model.fit(train_X, train_y, validation_split=0.2, epochs=300, callbacks=[early_stopping_monitor, NeptuneMonitor()])
else:
   model.fit(train_X, train_y, validation_split=0.2, epochs=300, callbacks=[early_stopping_monitor])

#save the model in neptune
if os.getenv('CI') == "true":
    model.save('model.h5')
    neptune.log_artifact('model.h5')

# evaluate the keras model
loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print('Test accuracy: %.2f\n' % (accuracy*100))
print('Test loss: %.2f\n' % (loss*100))
if os.getenv('CI') == "true":
    neptune.log_metric('Test accuracy', accuracy*100)
    neptune.log_metric('Test loss', loss*100)
# Log metrics to Neptune

# Handle CI pipeline details
if os.getenv('CI') == "true":
    neptune.append_tag('ci-pipeline', os.getenv('NEPTUNE_EXPERIMENT_TAG_ID'))

