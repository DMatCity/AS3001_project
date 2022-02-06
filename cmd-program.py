import sys
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
from functions import * 

# Where to save the figures
PROJECT_ROOT_DIR = '{}/02_Output'.format(sys.path[0])
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Plots")
os.makedirs(IMAGES_PATH, exist_ok=True)

# Define some functions for callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
#tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Define n steps taken into account to predict t steps into the future  
n_steps = 250
t_steps = 60

# Define a list with then names of all the files to be read
input_files = ['AAPL.csv','GE.csv','JPM.csv','DAL.csv','WMT.csv']

# exract names without the extentsion
files = []
for i in range(len(input_files)):
    files.append(input_files[i].partition('.')[0])

# output list valuation set
df_output_val = pd.DataFrame(columns=['Linear Regr', 'LSTM', 'CNN-LSTM','WaveNet'], index=files)

# output list test set
df_output_test = pd.DataFrame(columns=['Linear Regr', 'LSTM', 'CNN-LSTM','WaveNet'], index=files)

for i in range(len(input_files)):
    # Set the directory to where the files are located
    new_dir = '{}/01_Data'.format(sys.path[0])
    new_dir = str(new_dir)
    os.chdir(new_dir)

    # Import the data
    db = read_data(input_files[i])
    # Export a statistical summary of the data 
    db.describe().to_csv('{}/02_Output/Tables/Description of {}.csv'.format(sys.path[0], files[i]))

    # Create a histogram to inspect the distribution of the data
    plt.hist(db, density=True)
    save_fig('Histogram before Log Transformation of {}'.format(files[i]), IMAGES_PATH)
    plt.clf()

    # Perfrom a  log transformation of the data to check if the closing prices look closer to a normal distribution
    db_log = np.log(db)
    # Export a summary statistic of the log tranformed data to compare to the original data set
    db_log.describe().to_csv('{}/02_Output/Tables/Description of log {}.csv'.format(sys.path[0], files[i]))

    # Create a histrogram to inspect the distribution of the log transformed data
    plt.hist(db_log, density=True)
    save_fig('Histogram after Log Transformation of {}'.format(files[i]), IMAGES_PATH)
    plt.clf()

    # Change the directory to where the plots are supposed to be saved
    os.makedirs(IMAGES_PATH, exist_ok=True)

    # Create the sets
    train, val, test = split_set(db_log)

    # Create and fit the scaler to the training data for a feature range of [0,1]
    # Transform the valuation and test set with the same parameters of the training set
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    # Prepare the actual sets to be feed into the machine learning models
    X_train, Y_train = preprocess_data(train,n_steps=n_steps, t_steps=t_steps)
    X_val, Y_val = preprocess_data(val,n_steps=n_steps, t_steps=t_steps)
    X_test, Y_test = preprocess_data(test,n_steps=n_steps, t_steps=t_steps)

    # Create Linear Regression Model as a baseline model to benchmark the performance of the others models
    X = np.array(range(1,n_steps + t_steps + 1))
    X = X.reshape(-1, 1)
    y = scaler.transform(np.array(db_log).reshape(-1,1))
    
    regr = linear_model.LinearRegression()
    mse = []
    for j in range(len(db_log) - n_steps-t_steps):
    
        regr.fit(X[:n_steps], y[j:j + n_steps])
        y_predict = regr.predict(X[-t_steps:])

        metrics = mean_squared_error(y[j+n_steps:j+n_steps + t_steps],y_predict)
        mse.append(metrics)

    average_mse = sum(mse)/len(mse)

    df_output_val.loc[files[i],'Linear Regr'] = average_mse
    df_output_test.loc[files[i],'Linear Regr'] = average_mse
    print(df_output_val, df_output_test)

    y = scaler.inverse_transform(y)
    y = np.exp(y)
    y_predict = scaler.inverse_transform(y_predict)
    y_predict = np.exp(y_predict)
    plot_series(X[-n_steps-t_steps:], n_steps)
    plt.plot(np.arange(n_steps + t_steps), y[-n_steps-t_steps:], ".-",color='lightskyblue', label="Actual")
    plt.plot(np.arange(n_steps, n_steps + t_steps), y_predict, "x-",color='orange', label="Forecast", markersize=5)
    plt.axis([n_steps - 5, n_steps + t_steps, 0, 200])
    plt.legend(fontsize=10)
    save_fig('Linear Regression {}'.format(files[i]), IMAGES_PATH)
    plt.clf()

    # Create LSTM Model
    def LSTM_model(unit, drop, lr):
        model = keras.models.Sequential([
            keras.layers.LSTM(unit, return_sequences=True, input_shape=[None, 1]),
            MCDropout(drop),
            keras.layers.LSTM(unit, return_sequences=True),
            MCDropout(drop),
            keras.layers.LSTM(unit, return_sequences=True),
            MCDropout(drop),
            keras.layers.TimeDistributed(keras.layers.Dense(t_steps))
            ])

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(loss="mse", optimizer=opt, metrics=[last_time_step_mse])
        return model

    # Create the search parameters for the bayesian optimization
    def build_model(hp):
        unit = hp.Int("unit", min_value=5, max_value=100, step=5, default=25)
        drop = hp.Float("drop", min_value=0.01, max_value=0.20, step=0.01, default=0.05)
        lr = hp.Float("lr", min_value=0.00001, max_value=0.01, sampling="log", default= 0.001)
        # call existing model-building code with the hyperparameter values.
        model = LSTM_model(unit, drop, lr)
        return model   

    build_model(kt.HyperParameters())

    # Create the bayesian optimization
    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_last_time_step_mse", direction="min"),
        max_trials=10,
        alpha=0.0001,
        beta=2.6,
        seed=None,
        overwrite=True,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
    )

    tuner.search_space_summary()
    # Perform the hyperparamter tunin process
    tuner.search(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))

    # Get the best model
    best_model = tuner.get_best_models()
    best_model = best_model[0]

    # Create a checkpoint to save the model
    checkpoint_cb = keras.callbacks.ModelCheckpoint('LSTM_{file}'.format(file=files[i]), save_best_only=True)
    best_model.fit(X_train, Y_train, epochs=1000, validation_data=(X_val, Y_val), callbacks=[ early_stopping_cb, checkpoint_cb])
    
    # Check if already a model exists, and if so, load the model and compile it. Otherwise, save the current model.
    if os.path.exists('{path}/02_Output/Models/LSTM_{file}'.format(path=sys.path[0],file=files[i])) == True:
        best_model = keras.models.load_model('{path}/02_Output/Models/LSTM_{file}'.format(path=sys.path[0],file=files[i]), custom_objects={"MCDropout": MCDropout, "LastTimeStepMSE": last_time_step_mse}, compile=False)
        best_model.compile(loss="mse", optimizer='adam', metrics=[last_time_step_mse])
        print('loading model...')
    else:
        best_model.save('{path}/02_Output/Models/LSTM_{file}'.format(path=sys.path[0],file=files[i]))
        print('saving model...')
    
    # Extract the perfomance metrics on the model and store it in a database which will be exported later to a csv file.
    loss, metrics = best_model.evaluate(X_val, Y_val)
    df_output_val.loc[files[i],'LSTM'] = metrics
    loss, metrics = best_model.evaluate(X_test, Y_test)
    df_output_test.loc[files[i],'LSTM'] = metrics

    # Print the model
    best_model.summary()
    
    # Output the current database of the metrics
    print(df_output_val, df_output_test)

    # Use the model to make predicton
    y_pred = best_model.predict(X_test[-1:,:,:])[:, -1][..., np.newaxis]
    y_plot = Y_test[-1:,-1:,:]
    y_plot = y_plot.reshape(y_plot.shape[0], y_plot.shape[2],y_plot.shape[1])

    # Revert the scaler and log transformation
    y_pred = scaler.inverse_transform(y_pred[:,:,0])
    y_pred = np.exp(y_pred)
    Y_plot = scaler.inverse_transform(y_plot[:,:,0])
    Y_plot = np.exp(Y_plot)
    X_plot = scaler.inverse_transform(X_test[-1:,:,0])
    X_plot = np.exp(X_plot)

    # Visualise a predicition sequence
    plot_multiple_forecasts_2D(X_plot, Y_plot, y_pred)
    save_fig('LSTM Forecast {}'.format(files[i]),IMAGES_PATH)
    plt.clf()

    # free up RAM and clear any values
    tf.keras.backend.clear_session()
    
    # Create the CNN-LSTM Model
    def CNN_LSTM_model(filters, unit, lr):
        model = keras.models.Sequential([
            keras.layers.Conv1D(filters=filters, kernel_size=4, strides=2, padding="valid", input_shape=[None, 1]),
            keras.layers.LSTM(unit, return_sequences=True),
            keras.layers.LSTM(unit, return_sequences=True),
            keras.layers.LSTM(unit, return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(t_steps))
        ])

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(loss="mse", optimizer=opt, metrics=[last_time_step_mse])
        return model

    def build_model(hp):
        filters = hp.Int("filters", min_value=10, max_value=100, step=5, default=t_steps)
        unit = hp.Int("unit", min_value=5, max_value=100, step=5, default=25)
        lr = hp.Float("lr", min_value=0.00001, max_value=0.01, sampling="log", default=0.001)
        # call existing model-building code with the hyperparameter values.
        model = CNN_LSTM_model(filters, unit, lr)
        return model   

    build_model(kt.HyperParameters())

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_last_time_step_mse", direction="min"),
        max_trials=10,
        alpha=0.0001,
        beta=2.6,
        seed=None,
        overwrite=True,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
    )

    tuner.search_space_summary()
    tuner.search(X_train, Y_train[:, 3::2], epochs=5, validation_data=(X_val, Y_val[:, 3::2]))

    # Get the top model
    best_model = tuner.get_best_models()
    best_model = best_model[0]

    checkpoint_cb = keras.callbacks.ModelCheckpoint('CNN-LSTM_{file}'.format(file=files[i]), save_best_only=True)
    best_model.fit(X_train, Y_train[:, 3::2], epochs=1000, validation_data=(X_val, Y_val[:, 3::2]), callbacks=[early_stopping_cb, checkpoint_cb])
    
    if os.path.exists('{path}/02_Output/Models/CNN-LSTM_{file}'.format(path=sys.path[0],file=files[i])) == True:
        best_model = keras.models.load_model('{path}/02_Output/Models/CNN-LSTM_{file}'.format(path=sys.path[0], custom_objects={"MCDropout": MCDropout, "LastTimeStepMSE": last_time_step_mse}, file=files[i]), compile=False)
        best_model.compile(loss="mse", optimizer='adam', metrics=[last_time_step_mse])
        print('loading model...')
    else:
        best_model.save('{path}/02_Output/Models/CNN-LSTM_{file}'.format(path=sys.path[0],file=files[i]))
        print('saving model')
    
    loss, metrics = best_model.evaluate(X_val, Y_val[:, 3::2])
    df_output_val.loc[files[i],'CNN-LSTM'] = metrics
    loss, metrics = best_model.evaluate(X_test, Y_test[:, 3::2])
    df_output_test.loc[files[i],'CNN-LSTM'] = metrics

    best_model.summary()
   
    y_pred = best_model.predict(X_test[-1:,:,:])[:, -1][..., np.newaxis]
    y_plot = Y_test[-1:,-1:,:]
    y_plot = y_plot.reshape(y_plot.shape[0], y_plot.shape[2],y_plot.shape[1])

    y_pred = scaler.inverse_transform(y_pred[:,:,0])
    y_pred = np.exp(y_pred)
    Y_plot = scaler.inverse_transform(y_plot[:,:,0])
    Y_plot = np.exp(Y_plot)
    X_plot = scaler.inverse_transform(X_test[-1:,:,0])
    X_plot = np.exp(X_plot)

    plot_multiple_forecasts_2D(X_plot, Y_plot, y_pred)
    save_fig('CNN-LSTM Forecast {}'.format(files[i]),IMAGES_PATH)
    plt.clf()

    keras.backend.clear_session()

    # Create the WaveNet model
    def WaveNet_model(filters, kernel, lr):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None, 1]))
        for rate in (1, 2, 4, 8) * 2:
            model.add(keras.layers.Conv1D(filters=filters, kernel_size=kernel, padding="causal", activation="relu", dilation_rate=rate))
        model.add(keras.layers.Conv1D(filters=t_steps, kernel_size=1))

        opt = keras.optimizers.Adam(lr=lr)
        model.compile(loss="mse", optimizer=opt, metrics=[last_time_step_mse])
        return model
    
    def build_model(hp):
        filters = hp.Int("filters", min_value=10, max_value=100, step=5, default=t_steps)
        kernel = hp.Int("kernel", min_value=1, max_value=10, step=1, default=4)
        lr = hp.Float("lr", min_value=0.00001, max_value=0.01, sampling="log", default=0.001)
        # call existing model-building code with the hyperparameter values.
        model = WaveNet_model(filters, kernel, lr)
        return model   

    build_model(kt.HyperParameters())

    tuner = kt.BayesianOptimization(
        hypermodel=build_model,
        objective=kt.Objective("val_last_time_step_mse", direction="min"),
        max_trials=10,
        alpha=0.0001,
        beta=2.6,
        seed=None,
        overwrite=True, 
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
    )

    tuner.search_space_summary()
    tuner.search(X_train, Y_train, epochs=5, validation_data=(X_val, Y_val))

    best_model = tuner.get_best_models()
    best_model = best_model[0]

    checkpoint_cb = keras.callbacks.ModelCheckpoint('WaveNets_{file}'.format(file=files[i]), save_best_only=True)
    best_model.fit(X_train, Y_train, epochs=1000, validation_data=(X_val, Y_val), callbacks=[early_stopping_cb, checkpoint_cb])
    

    if os.path.exists('{path}/02_Output/Models/WaveNets_{file}'.format(path=sys.path[0],file=files[i])) == True:
        best_model = keras.models.load_model('{path}/02_Output/Models/WaveNets_{file}'.format(path=sys.path[0],file=files[i]), compile=False)
        best_model.compile(loss="mse", optimizer='adam', metrics=[last_time_step_mse])
        print('loading model...')
    else:
        best_model.save('{path}/02_Output/Models/WaveNets_{file}'.format(path=sys.path[0],file=files[i]))
        print('saving model...') 

    
    loss, metrics = best_model.evaluate(X_val, Y_val)
    df_output_val.loc[files[i],'WaveNet'] = metrics
    loss, metrics = best_model.evaluate(X_test, Y_test)
    df_output_test.loc[files[i],'WaveNet'] = metrics


    best_model.summary()
    
    y_pred = best_model.predict(X_test[-1:,:,:])[:, -1][..., np.newaxis]
    y_plot = Y_test[-1:,-1:,:]
    y_plot = y_plot.reshape(y_plot.shape[0], y_plot.shape[2],y_plot.shape[1])

    y_pred = scaler.inverse_transform(y_pred[:,:,0])
    y_pred = np.exp(y_pred)
    Y_plot = scaler.inverse_transform(y_plot[:,:,0])
    Y_plot = np.exp(Y_plot)
    X_plot = scaler.inverse_transform(X_test[-1:,:,0])
    X_plot = np.exp(X_plot)

    plot_multiple_forecasts_2D(X_plot, Y_plot, y_pred)
    save_fig('WaveNet not Original {}'.format(files[i]),IMAGES_PATH)
    plt.clf()

    keras.backend.clear_session()

print(df_output_val, df_output_test)

df_output_val.to_csv('{}/02_Output/Tables/Table_of_MSE_val.csv'.format(sys.path[0]))
df_output_test.to_csv('{}/02_Output/Tables/Table_of_MSE_test.csv'.format(sys.path[0]))


