from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def save_fig(fig_id, Images_path, tight_layout=False, fig_extension="pdf", resolution=300):
    path = os.path.join(Images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_series(series, n_steps, y=None, y_pred=None, x_label="Trading Days", y_label="Closing Price", legend=True):
    plt.plot(series,".-",color='steelblue')
    if y is not None:
        plt.plot(n_steps, y, ".-",color='lightskyblue', label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "x-",color='orange', markersize=5, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=10)
    if y_label:
        plt.ylabel(y_label, fontsize=10)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, 0, 1])
    if legend and (y or y_pred):
        plt.legend(fontsize=10, loc="upper left")

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def plot_multiple_forecasts_3D(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0], n_steps)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], ".-",color='lightskyblue', label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "x-",color='orange', label="Forecast", markersize=5)
    plt.axis([n_steps - 5, n_steps + ahead, 0, 200])
    plt.legend(fontsize=10)

def plot_multiple_forecasts_3D_test(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[2]
    iterations = [0,60,120,180]
    plot_series(X[0, :, 0], n_steps)
    for i in iterations:
        plt.plot(np.arange(n_steps + i, n_steps + ahead + i), Y[i, -1, :], ".-",color='lightskyblue', label="Actual")
        plt.plot(np.arange(n_steps + i, n_steps + ahead + i), Y_pred[i, :, :], "x-",color='orange', label="Forecast", markersize=5)
    plt.axis([n_steps - 5, 500, 0, 2])
    plt.legend(fontsize=10)

def plot_multiple_forecasts_2D(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :], n_steps)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :], ".-",color='lightskyblue', label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :], "x-",color='orange', label="Forecast", markersize=5)
    plt.axis([n_steps - 5, n_steps + ahead, 0, 200])
    plt.legend(fontsize=10)

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

def create_sets_vector(db_scaled, n_steps, t_steps):
    new_series = []
    for row in range(len(db_scaled)-n_steps-t_steps+1):
        ser = db_scaled[row:row+n_steps+t_steps]
        new_series.append(ser)
    
    new_series = np.array(new_series)
    new_series = new_series.reshape(new_series.shape[0], new_series.shape[1], 1)

    len_set = len(new_series)
    len_train = int(2/3 * len_set)
    len_val = int(5/6 * len_set)
    len_test = len_set - 1

    X_train, Y_train = new_series[:len_train, :n_steps], new_series[:len_train,-t_steps:-1,0]
    X_val, Y_val = new_series[len_train:len_val, :n_steps], new_series[len_train:len_val, -t_steps:,0]
    X_test, Y_test = new_series[len_val:len_test, :n_steps], new_series[len_val:len_test, -t_steps:,0]
    X_new, Y_new = new_series[-1:,:n_steps], new_series[len_test:,-t_steps:,0]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_new, Y_new

def split_set(db):
    len_set = len(db)
    len_train = int(2/3 * len_set)
    len_val = int(5/6 * len_set)

    train, val, test = db[:len_train], db[len_train:len_val], db[len_val:]
    train = np.array(train).reshape(-1,1)
    val = np.array(val).reshape(-1,1)
    test = np.array(test).reshape(-1,1)
    return train, val, test

def preprocess_data(db_scaled,n_steps,t_steps):
    new_series = []
    for row in range(len(db_scaled)-n_steps-t_steps+1):
        ser = db_scaled[row:row+n_steps+t_steps]
        new_series.append(ser)
    
    new_series = np.array(new_series)
    new_series = new_series.reshape(new_series.shape[0], new_series.shape[1], 1)

    len_set = len(new_series)

    Y = np.empty((len_set, n_steps, t_steps))
    for step_ahead in range(1, t_steps + 1):
        Y[ : , : ,step_ahead - 1] = new_series[ : , step_ahead:step_ahead + n_steps, 0]
        
    X = new_series[:,:n_steps]

    return  X, Y

def create_sets_sequence_sc(db, n_steps, t_steps):
    
    len_set = len(db)
    len_train = int(2/3 * len_set)
    len_val = int(5/6 * len_set)
    len_test = len_set

    print(len_set)
    print(len_train)
    print(len_val)
    print(len_test)

    train = db[:len_train]
    val = db[len_train:len_val]
    test = db[len_val:len_test]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    #train, mean_train, std_train = transf_data(train)
    #val, mean_val, std_val = transf_data(val)
    #test, mean_test, std_test = transf_data(test)
    
    def preprocess_data(db_scaled):
        new_series = []
        for row in range(len(db_scaled)-n_steps-t_steps+1):
            ser = db_scaled[row:row+n_steps+t_steps]
            new_series.append(ser)
    
        new_series = np.array(new_series)
        new_series = new_series.reshape(new_series.shape[0], new_series.shape[1], 1)

        len_set = len(new_series)

        Y = np.empty((len_set, n_steps, t_steps))
        for step_ahead in range(1, t_steps + 1):
            Y[ : , : ,step_ahead - 1] = new_series[ : , step_ahead:step_ahead + n_steps, 0]
        
        X = new_series[:,:n_steps]

        return  X, Y

    X_train, Y_train = preprocess_data(train)
    X_val, Y_val = preprocess_data(val)
    X_test, Y_test = preprocess_data(test)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, #mean_test, #std_test

def create_sets_sequence(db_scaled, n_steps, t_steps):
    new_series = []
    for row in range(len(db_scaled)-n_steps-t_steps+1):
        ser = db_scaled[row:row+n_steps+t_steps]
        new_series.append(ser)
    
    new_series = np.array(new_series)
    new_series = new_series.reshape(new_series.shape[0], new_series.shape[1], 1)

    len_set = len(new_series)
    len_train = int(2/3 * len_set)
    len_val = int(5/6 * len_set)
    len_test = len_set - 1

    Y = np.empty((len_set, n_steps, t_steps))
    for step_ahead in range(1, t_steps + 1):
        Y[ : , : ,step_ahead - 1] = new_series[ : , step_ahead:step_ahead + n_steps, 0]

    X_train, Y_train = new_series[:len_train, :n_steps], Y[:len_train]
    X_val, Y_val = new_series[len_train:len_val, :n_steps], Y[len_train:len_val]
    X_test, Y_test = new_series[len_val:len_test, :n_steps], Y[len_val:len_test]
    X_new, Y_new = new_series[-1:,:n_steps], new_series[len_test:,n_steps:]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_new, Y_new

def read_data(file_input):
    dataset = pd.read_csv(file_input, skiprows=1)
    dataset.columns = ['Date', 'Close']
        
    dataset = dataset.dropna(how='any')
    
    db = dataset['Close']
    return db

def transf_data(db):
    #db = np.log(db) / np.array(10)
    #return db
    db = np.log(db)
    mean_db = np.mean(db)
    std_db = np.std(db)
    new_db = (db - np.array(mean_db)) / np.array(std_db)

    return new_db, mean_db, std_db

def inv_tranf_data(output, mean_db, std_db):
    #output = np.array(10) * output
    #output = np.exp(output)
    #return output

    output = output * np.array(std_db)
    output = output + mean_db
    output = np.exp(output)

    return output

def inv_tranf_data_sc(output, mean_db, std_db):
    
    
    pass

class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
    def get_config(self):
        return {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
         }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GatedActivationUnit(keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate

def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal",
                            dilation_rate=dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z


