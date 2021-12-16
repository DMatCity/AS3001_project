import sys
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from functions import * 

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "plots", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Define some functions for callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint('my_keras_model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


# Define n steps taken into account to predict t steps into the future  
n_steps = 250
t_steps = 60

# Define a list with then names of all the files to be read
input_files = ['AAPL.csv','GE.csv','JPM.csv','DAL.csv','WMT.csv']

# output list
df_output = pd.DataFrame(columns=['Linear Regr', 'LSTM', 'CNN-LSTM','WaveNet'], index=input_files)

for i in range(len(input_files)):
    # Set the directory to where the files are located
    new_dir = '{}/01_Data'.format(sys.path[0])
    #print(new_dir)
    new_dir = str(new_dir)
    os.chdir(new_dir)

    db = read_data(input_files[i])  
    #print(db)

    os.makedirs(IMAGES_PATH, exist_ok=True)

    #db_AAPL = db_AAPL.reshape(db_AAPL.shape[0], db_AAPL.shape[1], 1)
    #X_train, Y_train, X_val, Y_val, X_test, Y_test, X_new, Y_new = create_sets_vector(db_AAPL, n_steps, t_steps)
    #len_set = len(db_AAPL)
    #X_train, Y_train = np.array(range(1,int(len_set * 2/3)+1)), db_AAPL[:int(len_set*2/3)]
    #X_val, Y_val = np.array(range(int(len_set * 2/3),int(len_set * 5/6))), db_AAPL[int(len_set*2/3):int(len_set*5/6)]
    #X_test, Y_test = np.array(range(int(len_set * 5/6), len_set)), db_AAPL[int(len_set*5/6):len_set +1]

    # Create Linear Regression Model as a baseline model to benchmark the performance of the others models
    X = np.array(range(1,len(db)+1))
    X = X.reshape(-1, 1)
    y = db

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    #print(results)
    #print(regr.coef_[0])
    #print(regr.intercept_)
    y_predict = regr.predict(X)

    metrics = mean_squared_error(y,y_predict)
    df_output.loc[input_files[i],'Linear Regr'] = metrics
    print(df_output)
    plt.scatter(X, y)
    plt.plot(X, y_predict)
    save_fig('Linear Regression {}'.format(input_files[i]), IMAGES_PATH)
    plt.clf()

    db = transf_data(db)
    # create train, val, test and new set
    X_train, Y_train, X_val, Y_val, X_test, Y_test, X_new, Y_new = create_sets_sequence(db, n_steps, t_steps)

    # Create LSTM Model
    model = keras.models.Sequential([
        keras.layers.LSTM(60, return_sequences=True, input_shape=[None, 1]),
        MCDropout(0.05),
        keras.layers.LSTM(60, return_sequences=True),
        MCDropout(0.05),
        keras.layers.TimeDistributed(keras.layers.Dense(t_steps))
    ])

    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss="mse", optimizer=opt, metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train, epochs=1000,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping_cb])

    loss, metrics = model.evaluate(X_test, Y_test)
    df_output.loc[input_files[i],'LSTM'] = metrics
    print(df_output)
    Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

    Y_pred = inv_tranf_data(Y_pred)
    Y_new = inv_tranf_data(Y_new)
    X_new = inv_tranf_data(X_new)

    plot_multiple_forecasts(X_new, Y_new, Y_pred)
    save_fig('LSTM Forecast {}'.format(input_files[i]),IMAGES_PATH)
    plt.clf()
    # free up RAM and clear any values
    tf.keras.backend.clear_session()
    # CNN-LSTM Model

    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=10, kernel_size=4, strides=2, padding="valid",
                            input_shape=[None, 1]),
        keras.layers.LSTM(60, return_sequences=True),
        keras.layers.LSTM(60, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(t_steps))
    ])

    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train[:, 3::2], epochs=1000,
                        validation_data=(X_val, Y_val[:, 3::2]), callbacks=[early_stopping_cb])

    loss, metrics = model.evaluate(X_test, Y_test[:, 3::2])
    df_output.loc[input_files[i],'CNN-LSTM'] = metrics
    
    X_new = transf_data(X_new)
    Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

    Y_pred = inv_tranf_data(Y_pred)
    #Y_new = inv_tranf_data(Y_new)
    X_new = inv_tranf_data(X_new)

    plot_multiple_forecasts(X_new, Y_new, Y_pred)
    save_fig('CNN-LSTM Forecast {}'.format(input_files[i]),IMAGES_PATH)
    plt.clf()
    # WaveNet Architecture
    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters=t_steps, kernel_size=4, padding="causal",
                                      activation="relu", dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=t_steps, kernel_size=1))
    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    history = model.fit(X_train, Y_train, epochs=1000,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping_cb])

    loss, metrics = model.evaluate(X_val, Y_val)
    df_output.loc[input_files[i],'WaveNet'] = metrics
    #series = generate_time_series(1, n_steps + t_steps)
    X_new = transf_data(X_new)
    Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

    Y_pred = inv_tranf_data(Y_pred)
    #Y_new = inv_tranf_data(Y_new)
    X_new = inv_tranf_data(X_new)

    plot_multiple_forecasts(X_new, Y_new, Y_pred)
    save_fig('WaveNet not Original {}'.format(input_files[i]),IMAGES_PATH)
    plt.clf()

print(df_output)
'''
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

n_layers_per_block = 3 # 10 in the paper
n_blocks = 2 # 3 in the paper
n_filters = 60 # 128 in the paper
n_outputs = 60 # 256 in the paper

inputs = keras.layers.Input(shape=[None, 1])
z = keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal")(inputs)
skip_to_last = []
for dilation_rate in [2**i for i in range(n_layers_per_block)] * n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
z = keras.activations.relu(keras.layers.Add()(skip_to_last))
z = keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)
Y_proba = keras.layers.Conv1D(n_outputs, kernel_size=1, activation="softmax")(z)

model = keras.models.Model(inputs=[inputs], outputs=[Y_proba])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=5,
                    validation_data=(X_val, Y_val))


Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

Y_pred = inv_tranf_data(Y_pred)
Y_new = inv_tranf_data(Y_new)
X_new = inv_tranf_data(X_new)

plot_multiple_forecasts(X_new, Y_new, Y_pred)
save_fig('WaveNet {}'.format(db_list[0]),IMAGES_PATH)
plt.show()
'''
