from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, Concatenate, Softmax 
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from model.DendralNeuron import DendralNeuron
import numpy as np

def generate_hyperparam_grid_Dendral_1_layer( min_num_of_neuron_x_layer, max_num_of_neuron_x_layer, num_of_trials, min_LR, max_LR ):

    hnn_models = np.random.randint(low = min_num_of_neuron_x_layer, high= max_num_of_neuron_x_layer, size=num_of_trials)
    LR_arr = np.random.uniform(low = min_LR, high= max_LR, size=num_of_trials)
    
    return hnn_models, LR_arr

def generate_hyperparam_grid( min_num_of_layers, max_num_of_layers, max_num_of_neuron_x_layer, num_of_trials, min_LR, max_LR ):

    num_of_layers = np.random.random_integers(low = min_num_of_layers, high = max_num_of_layers, size = num_of_trials )

    dnn_models = []
    neurons_per_layer = []

    for n_layer in num_of_layers :
        for layer in range( n_layer ):
            
                if layer== 0 :
                    neurons = np.random.random_integers( low = int(max_num_of_neuron_x_layer /2) , high = max_num_of_neuron_x_layer)
                    neurons_top = neurons
                else:
                    neurons = np.random.random_integers( low = int(neurons_top /2),  high = neurons_top )
                    if ( neurons == 0):
                        neurons =1
    
                    neurons_top = neurons
                     
                neurons_per_layer.append( neurons)
            
                
        dnn_models.append( neurons_per_layer )
        neurons_per_layer = []
    
    
    LR_arr = np.random.uniform(low= min_LR, high= max_LR, size=num_of_trials)
    
    return dnn_models, LR_arr

    
def build_MLNN( morph_neurons, activation, input_shape, num_classes, LR):
    
    model = Sequential()

    model.add(DendralNeuron(morph_neurons, activation= activation, input_shape=input_shape))
    
    if ( num_classes > 2):
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(num_classes, activation='sigmoid'))
    
    model.compile(loss = "categorical_crossentropy",
                  optimizer= Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
        
    return model

def train_MLNN(model,  x_train, y_train, batch_size, epochs, x_test, y_test, nb_verbose):
    
    hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = False,
          validation_data=(x_test, y_test))
    
    return hist    
