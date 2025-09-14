import sys
import random
import numpy as np
import tensorflow as tf
from model import BuildModel as bm
from utils import Utils_Model as um
from Plot import PlotUtils as pltU
from Datasource import DatasourceUtils as ldts
from callback.CustomCallback import CustomCallback 
import math

epoch_callback_weigths = []

def enviroment():
    print(f"Enviroment: ")
    print(f"\tPython {sys.version}")
    print(f"\tNumpy  {np.__version__}")
    print(f"\tTensor Flow Version: {tf.__version__}")
    print(f"\tKeras Version: {tf.keras.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("\tGPU is", "available" if gpu else "NOT AVAILABLE")
    
    print(f"\tEager execution: {tf.executing_eagerly()} ")
    print("-----------------------------------------------------------------")
    
def reproducibility():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    
def classify_DataSet_Spiral( dendral_neurons =  402, #value on paper
                        lr = 0.009922894436983054,
                        activation = 'tanh',
                        v_verbose = False):

    #load Dataset 
    name_file = "Spiral_2_loop_1"
    n_class = 2
    n_loops = 1
    plot= True 
    normalize = False
    train_dataset, test_dataset, P_T,  Ptest_Ttest, input_dims = ldts.loadDataset_Espiral_NClass_N_Loops ( n_class = n_class, n_loops = n_loops, plot= plot, normalize = False )
    

    input_shape = (P_T[0].shape[1],) 
        
    output_shape = P_T[1].shape[1] 
    
    batch_size = 128
    nb_epoch = 250
    
    #MLNN with OIoU-Metric 
    loss_type = 'Normal'
    model = bm.build_HybridModel_DN_MLP( dendral_neurons, activation, input_shape,
                                         output_shape,path_save=None, regularize = False,
                                         dropout= 0 )
    
    [hist_OIoU_Metric, _] = bm.train_HybridModel_DN_MLP( model, lr, P_T[0], P_T[1],
                                                      Ptest_Ttest[0], Ptest_Ttest[1], 
                                                      batch_size, nb_epoch, v_verbose,
                                                      modelChkPnt= CustomCallback(),
                                                      loss_type= loss_type )
    
    best =  np.argmax( hist_OIoU_Metric.history['val_accuracy'] )    
    b_val_acc = hist_OIoU_Metric.history['val_accuracy'] [best]
        
    iou_best_hist = np.asanyarray( epoch_callback_weigths)
    
    max_rate_reduction = iou_best_hist[best]/100.0

    print(f" Dataset {name_file} ")
    print(f" Classification rate: {b_val_acc}")
    print(f" OIoU Metric        : {iou_best_hist[best]}")
    
    [b_nb_neurons, b_val_acc, new_OIoU] = algorithm_1(lr, loss_type, dendral_neurons, b_val_acc, max_rate_reduction,
                activation, input_shape, output_shape, P_T, Ptest_Ttest,
                batch_size, nb_epoch, v_verbose, iou_best_hist[best], b_oiu_lr=0 )
    
    algorithm_2(lr, b_nb_neurons, b_val_acc, activation, input_shape, output_shape,
                P_T, Ptest_Ttest, batch_size, nb_epoch, v_verbose, loss_type)
    
    
def classify_DataSet_XOR( dendral_neurons =  320, #value on paper
                        lr = 0.009922894436983054,
                        activation = 'tanh',
                        v_verbose = False):

     #load Dataset XOR
    name_file = "DataSet_XOR"
    train_dataset, test_dataset, P_T, Ptest_Ttest, input_dims = ldts.loadDataset_XOR()

    input_shape = (P_T[0].shape[1],) 
        
    output_shape = P_T[1].shape[1] 
    
    batch_size = 128
    nb_epoch = 250
    
    #MLNN with OIoU-Metric 
    loss_type = 'Normal'
    model = bm.build_HybridModel_DN_MLP( dendral_neurons, activation, input_shape,
                                         output_shape,path_save=None, regularize = False,
                                         dropout= 0 )
    
    [hist_OIoU_Metric, _] = bm.train_HybridModel_DN_MLP( model, lr, P_T[0], P_T[1],
                                                      Ptest_Ttest[0], Ptest_Ttest[1], 
                                                      batch_size, nb_epoch, v_verbose,
                                                      modelChkPnt= CustomCallback(),
                                                      loss_type= loss_type )
    
    best =  np.argmax( hist_OIoU_Metric.history['val_accuracy'] )    
    b_val_acc = hist_OIoU_Metric.history['val_accuracy'] [best]
        
    iou_best_hist = np.asanyarray( epoch_callback_weigths)
    
    max_rate_reduction = iou_best_hist[best]/100.0

    print(" Dataset X-OR ")
    print(f" Classification rate: {b_val_acc}")
    print(f" OIoU Metric        : {iou_best_hist[best]}")
    
    [b_nb_neurons, b_val_acc, new_OIoU] = algorithm_1(lr, loss_type, dendral_neurons, b_val_acc, max_rate_reduction,
                activation, input_shape, output_shape, P_T, Ptest_Ttest,
                batch_size, nb_epoch, v_verbose, iou_best_hist[best], b_oiu_lr=0 )
    
    algorithm_2(lr, b_nb_neurons, b_val_acc, activation, input_shape, output_shape,
                P_T, Ptest_Ttest, batch_size, nb_epoch, v_verbose, loss_type)


def classify_DataSet_A( dendral_neurons =  393, #value on paper
                        lr = 0.009922894436983054,
                        activation = 'tanh',
                        v_verbose = False):

    name_file="DataSet_A"
    train_dataset, test_dataset, P_T,  Ptest_Ttest, input_dims = ldts.loadDataset_A()

    input_shape = (P_T[0].shape[1],) 
        
    output_shape = P_T[1].shape[1] 
    
    batch_size = 128
    nb_epoch = 250
    
    #MLNN with OIoU-Metric 
    loss_type = 'Normal'
    model = bm.build_HybridModel_DN_MLP( dendral_neurons, activation, input_shape,
                                         output_shape,path_save=None, regularize = False,
                                         dropout= 0 )
    
    [hist_OIoU_Metric, _] = bm.train_HybridModel_DN_MLP( model, lr, P_T[0], P_T[1],
                                                      Ptest_Ttest[0], Ptest_Ttest[1], 
                                                      batch_size, nb_epoch, v_verbose,
                                                      modelChkPnt= CustomCallback(),
                                                      loss_type= loss_type )
    
    best =  np.argmax( hist_OIoU_Metric.history['val_accuracy'] )    
    b_val_acc = hist_OIoU_Metric.history['val_accuracy'] [best]
        
    iou_best_hist = np.asanyarray( epoch_callback_weigths)
    
    max_rate_reduction = iou_best_hist[best]/100.0

    print(" Dataset A ")
    print(f" Classification rate: {b_val_acc}")
    print(f" OIoU Metric        : {iou_best_hist[best]}")
    
    [b_nb_neurons, b_val_acc, new_OIoU] = algorithm_1(lr, loss_type, dendral_neurons, b_val_acc, max_rate_reduction,
                activation, input_shape, output_shape, P_T, Ptest_Ttest,
                batch_size, nb_epoch, v_verbose, iou_best_hist[best], b_oiu_lr=0 )
    
    algorithm_2(lr, b_nb_neurons, b_val_acc, activation, input_shape, output_shape,
                P_T, Ptest_Ttest, batch_size, nb_epoch, v_verbose, loss_type)
    
    #MLNN with OIoU-Loss
        
     
def algorithm_1(lr, loss_type, b_nb_neurons, b_val_acc, max_rate_reduction,
                activation, input_shape, output_shape, P_T, Ptest_Ttest,
                batch_size, nb_epoch, v_verbose, Original_OIoU_metric, b_oiu_lr =0 # normal loss type
                ):
    
    num_of_customization_trials =0
    train_time_hist = []
    b_val_acc_Temp = b_val_acc
    b_nb_neurons_Temp = b_nb_neurons
    
    if max_rate_reduction == 0.0:
        num_of_customization_trials = 0
    else:
        num_of_customization_trials = 10
    
    if max_rate_reduction < 0.05:
        percentaje_of_custom_trials = 0.75
    else:
        percentaje_of_custom_trials = 0.25
        
    new_reduced_newurons = b_nb_neurons
    b_nb_neurons_ori = b_nb_neurons
    cont=0
    
    new_class_rate = b_val_acc
    while ( cont < num_of_customization_trials ):
        cont = cont +1
    
        new_reduced_newurons = math.floor( new_reduced_newurons - 
                                  new_reduced_newurons*(max_rate_reduction*percentaje_of_custom_trials))
    
        model = bm.build_HybridModel_DN_MLP( new_reduced_newurons, activation, input_shape, output_shape, path_save=None, regularize = False, dropout= 0 )
                
        [hist, train_time] = bm.train_HybridModel_DN_MLP( model, lr, P_T[0], P_T[1],
                                                      Ptest_Ttest[0], Ptest_Ttest[1], 
                                                      batch_size, nb_epoch, v_verbose,
                                                      modelChkPnt= CustomCallback(),
                                                      loss_type= loss_type,
                                                      LR_Iou = b_oiu_lr )
    
        train_time_hist.append(train_time)
        
        best =  np.argmax( hist.history['val_accuracy'] )    
        new_class_rate = hist.history['val_accuracy'] [best]

        if ( round(b_val_acc, 2) <= round(new_class_rate, 2)  ):
            b_hist = hist
            b_idx =  best
            b_lr  = lr
            b_batch_size = batch_size 
            b_nb_epoch = nb_epoch 
            b_model = model
            b_nb_neurons = new_reduced_newurons
            b_train_time = train_time
            b_val_acc = new_class_rate
            b_best = best
            iou_best_hist = np.asanyarray( epoch_callback_weigths)
            max_rate_reduction = epoch_callback_weigths[-1]/100.0
            
        print(f"Iteration: {cont}")
        print(f"\t ---> Training  ori_arqui:{b_nb_neurons_ori}\t-->new arqui: {new_reduced_newurons}  LR:  {lr}"  )
        print(f"\t ---> Original_clas: {b_val_acc}\t-->New_class:{new_class_rate} ")
        print(f"\t ---> Test Ori_Clas: {round(b_val_acc, 2)}\t -->New_class:{round(new_class_rate, 2)}\t ---> Condition {round(b_val_acc, 2) <= round(new_class_rate, 2)} ")
        print(f"\t ---> Reduction rate : {max_rate_reduction} ")
        
    print(f"-----------------------------------------------------------------")
    print(f"-----------------------------------------------------------------")
    print(f"-----------Output Algorithm 1 Reducing Network size --------------")
    print(f"\tOriginal architecture: {b_nb_neurons_Temp} --> new architecture: {b_nb_neurons}")
    print(f"\tOriginal class rate:   {b_val_acc}         --> new clas rate   : {b_val_acc_Temp}")
    print(f"\tOriginal OIoU: {Original_OIoU_metric}      --> new OIoU        : {iou_best_hist[b_idx]}")
    print(f"\n\n")
    
    return [ b_nb_neurons, b_val_acc, iou_best_hist[b_idx] ]

def algorithm_2(lr, b_nb_neurons, b_val_acc, activation, input_shape, output_shape,
                 P_T, Ptest_Ttest, batch_size, nb_epoch, v_verbose,  # normal loss type
                loss_type = 'IoU_Loss_LR'):
    
    LR_Iou = [ 0.001, 0.005 , 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    iou_model = []
    iou_hist  = []
    ioi_iou   = []
    iou_best_hist= []
    iou_best_hist.append(0)
    
    b_val_acc = -1
    b_idx = 0
    b_oiu_lr = 0
    
    for oiu_lr in  LR_Iou:
        
        model = bm.build_HybridModel_DN_MLP( b_nb_neurons, activation,
                                        input_shape, output_shape, 
                                        path_save=None, regularize = False,
                                        dropout= 0 )
                
        [hist, train_time] = bm.train_HybridModel_DN_MLP( model, lr, P_T[0], P_T[1],
                                                      Ptest_Ttest[0], Ptest_Ttest[1], 
                                                      batch_size, nb_epoch, v_verbose,
                                                      modelChkPnt= CustomCallback(),
                                                      loss_type= loss_type,
                                                      LR_Iou = oiu_lr )

        iou_model.append(model)
        iou_hist.append(hist)
        ioi_iou.append(epoch_callback_weigths)
        
        print(f"Iou Retraining:")
        best =  np.argmax( hist.history['val_accuracy'] )
        b_val_acc_it = hist.history['val_accuracy'] [best]
    
        if ( b_val_acc < b_val_acc_it  ):
            b_hist = hist
            b_idx =  best
            b_lr  =  lr
            b_batch_size = batch_size 
            b_nb_epoch = nb_epoch 
            b_model = model
            b_nb_neurons = b_nb_neurons
            b_train_time = train_time
            b_val_acc = b_val_acc_it
            b_best = best
            iou_best_hist = np.asanyarray( epoch_callback_weigths)
            b_oiu_lr = oiu_lr
            
        print(f"\t --->IoU Loss LR: {oiu_lr}" + 
              f" Last IoU: {epoch_callback_weigths[-1]} "+
              f" Val Acc: {hist.history['val_accuracy'][ best ]}" +
              f" Acc Train: {hist.history['accuracy'][best]} " + 
              f" LR: {b_lr}" + 
              f" batch_size: {b_batch_size}" + 
              f" nb_epoch: {b_nb_epoch}"  + 
              f" model_params: {model.count_params()}" +
              f" Architecture: {b_nb_neurons} " +
              f" Time: {train_time}" )
    
        #pltU.plot_stats( hist.history, hist_IoU_TF = epoch_callback_weigths,
        #                loss_type = loss_type, datasetName = name_file,
        #                save_fig = True,
        #                path_fig_save = "tmp_data/"+name_file+"/"+loss_type+"/"+str(oiu_lr)+"/stats"+str(oiu_lr)+".pdf" )
            
    print(f"-----------------------------------------------------------------")
    print(f"-----------------------------------------------------------------")
    print(f"-----------Output Algorithm 2  OIoU Loss --------------")
    print(f"\tOriginal Classification rate : {b_val_acc} --> new Classification rate: {hist.history['val_accuracy'][ best ]}")
    print(f"\tArchitecture:   {b_nb_neurons} ")
    print(f"\Best IoU_LR --> {b_oiu_lr}")
