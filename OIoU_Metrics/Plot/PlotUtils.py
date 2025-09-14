import os
import numpy as np
import matplotlib.pyplot as plt

def plot_stats( history = None, hist_IoU =None, hist_IoU_TF = None,
                datasetName = '', save_fig = False, 
                path_fig_save = None, loss_type = '' ):
    
    if ( "loss" in history.keys() ):
        x = list(range(len(history["loss"])))
        plt.plot(x, history["loss"], 'b', label='loss')
        
    if ( "Final_Loss" in history.keys() ):
        x = list(range(len(history["Final_Loss"])))
        plt.plot(x, history["Final_Loss"], 'b', label='FF_Loss')
        
    if ( "Dendral_Loss" in history.keys() ):
        y = list(range(len(history["Dendral_Loss"])))
        plt.plot(y, history["Dendral_Loss"], 'r', label='Dendral_Loss')
        
    if ( "Dense_Loss" in history.keys() ):
        z = list(range(len(history["Dense_Loss"])))
        plt.plot(z, history["Dense_Loss"], 'g', label='Dense_Loss')
        
    if not ( hist_IoU is None ):
        w_IoU = list(range(len(hist_IoU)))
        plt.plot(w_IoU, hist_IoU, 'pink', label="IoU_Weigths_numpy")
        
    if not ( hist_IoU_TF is None):
        w_IoU_TF = list(range(len(hist_IoU_TF)))
        plt.plot(w_IoU_TF, hist_IoU_TF, 'pink', label="IoU_Weigths_TF")
    
    
    plt.title('Training '+ loss_type +' loss: ' + datasetName)
        
    plt.legend()
    plt.grid( axis='both')
    
    if save_fig :
        directory = os.path.dirname(path_fig_save)
            
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig( path_fig_save )
        plt.close()

    else:
        plt.show()
    
def plot_espiral( P, num_class ):
    
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    
    dx, dy = (xmax - xmin)*0.1, (ymax - ymin)*0.1
    
   
    half = int(P.shape[0]/2)
    plt.scatter(P[0: half, 0],     P[0: half, 1]     , s=1)
    plt.scatter(P[half:half*2, 0], P[half:half*2, 1] , s=1)
    
    plt.grid( axis='both')
    plt.show()

def plot_N_class( P, T ):
    
    values_clases = np.unique(T)
    
    array_value_by_class = []
    
    for i in values_clases:
        array_value_by_class.append( P [np.where( T == i)[0]] )
    
    for arr in array_value_by_class:
        px = arr[:,0:2]
        plt.scatter(px[:, 0], px[:,1]     , s=1)
    
    plt.grid( axis='both')
    plt.show()
    print("\n Done ...")

def plot_decision_boundaries(model, X, Y, n_class, mesh_samples = 0, xpand= 0.1, plot= True, save_fig= False, file_path = None ):
    
    if(n_class == 2):        
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()
    
        dx, dy = (xmax - xmin)*xpand, (ymax - ymin)*xpand
 
        if ( mesh_samples == 0 ):
            mesh_samples = Y.shape[0]
            
        x = np.linspace(xmin-dx, xmax+dx, mesh_samples, dtype=np.float32)
        y = np.linspace(ymin-dy, ymax+dy, mesh_samples, dtype=np.float32)
        
        xx, yy = np.meshgrid( x, y ) 
        
        temp = np.c_[xx.ravel(), yy.ravel()]      
    
        z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        
        z = z.reshape((z.shape[0], z.shape[1]))
        
        zz = z.reshape(xx.shape)
        
        half = int(X.shape[0]/2)
        plt.scatter(X[0: half, 0],     X[0: half, 1]     , s=1)

        plt.scatter(X[half:half*2, 0], X[half:half*2, 1] , s=1)

        plt.grid( axis='both')
        
        plt.contour(xx, yy, zz, colors='k', levels=[0.50, 0.51])

    if(n_class == 3):
    
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), X[:, 1].max()
    
        dx, dy = (xmax - xmin)*xpand, (ymax - ymin)*xpand

        if ( mesh_samples == 0 ):
            mesh_samples = Y.shape[0]
            
        x = np.linspace(xmin-dx, xmax+dx, mesh_samples, dtype=np.float32)
        y = np.linspace(ymin-dy, ymax+dy, mesh_samples, dtype=np.float32)
        
        xx, yy = np.meshgrid( x, y ) 
                 
        z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0 )
        
        
        z = z.reshape((z.shape[0], z.shape[1]))
        
        z_temp = np.zeros( (z.shape[0], 1))
        
        for idx in range(0, z.shape[0]):
            z_temp[idx] = np.argmax(z[idx])
        
        zz = z_temp.reshape(xx.shape)
        
        values_clases = np.unique( Y )
        array_value_by_class = []
        
        for i in values_clases:
            array_value_by_class.append( X [np.where( Y == i)[0]] )
    
        for arr in array_value_by_class:
            px = arr[:,0:2]
            plt.scatter(px[:, 0], px[:,1]     , s=1)
        
        
        plt.grid( axis='both')
        
        plt.contour(xx, yy, zz, colors='k', levels=[0.50, 0.51])

        
    if save_fig:
        directory = os.path.dirname(file_path)
            
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        plt.savefig( file_path )
        plt.close()
    if plot:
            plt.show()

def plot_Normal_vs_OIou_Loss( dict_NormalLoss, dict_OIoULoss, dict_path_save):
    MEDIUM_SIZE = 12
    LEGEND_SIZE = 12 
    legend_size = 6
    y_axes_size = 12
    
    for normal_loss, OIoU_loss, path_save_key in zip(dict_NormalLoss.keys(),dict_OIoULoss.keys(), dict_path_save.keys() ):
    
        fig, ax1 = plt.subplots()
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        
        normal_loss_vals = dict_NormalLoss[normal_loss]
        dict_oiou_loss   = dict_OIoULoss[OIoU_loss]
        path_save        = dict_path_save [path_save_key]

        x1_vals = list(range(normal_loss_vals.shape[0]))

        if normal_loss in ['IMDB_10_Normal', 'IMDB_25_Normal', 'MNIST_Normal'] :           
           x1_vals = list(range(250))
           temp = np.zeros_like( x1_vals )
           ax1.plot(x1_vals, temp, label=normal_loss, linestyle='solid', color="black")           
        else:
           ax1.plot(x1_vals, normal_loss_vals, label=normal_loss, linestyle='solid', color="black")
        
        for lr in dict_oiou_loss.keys():
            lr_loss_vals = dict_oiou_loss[lr]
            x1_vals = list(range(lr_loss_vals.shape[0]))
            
            ax1.plot(x1_vals, lr_loss_vals, label=OIoU_loss+"_"+lr, linestyle='dashed', alpha=0.5)
            
        ax1.grid()
    
        plt.title( "Normal Loss vs OIoU-Metric Loss " + normal_loss )
        plt.legend(  fontsize = LEGEND_SIZE, loc='upper left')
        fig.tight_layout()  
    
        from matplotlib.pyplot import xlim
        xlim(right=250)  
        xlim(left=0)  
        
        plt.legend(  fontsize = legend_size)
        plt.savefig(path_save+"losses_"+normal_loss+".pdf")
        plt.show()

def plot_Normal_IoU_Metric(dict):
    
    MEDIUM_SIZE = 12
    LEGEND_SIZE = 12 
    fig, ax1 = plt.subplots()
    plt.rc('axes', labelsize=MEDIUM_SIZE)    
    
    for normal_metric in dict.keys():
        normal_IoU_Metric =dict[normal_metric]

        x1_vals = list(range(normal_IoU_Metric.shape[0]))

        if normal_metric in ['IMDB_10_Normal', 'IMDB_25_Normal', 'MNIST_Normal'] :
           
           x1_vals = list(range(250))
           temp = np.zeros_like( x1_vals )
           ax1.plot(x1_vals, temp, label=normal_metric)
           
        else:
            ax1.plot(x1_vals, normal_IoU_Metric, label=normal_metric)
        
    
    ax1.grid()
    
    plt.title( "OIoU - Metric"  )
    plt.legend(  fontsize = LEGEND_SIZE, loc='upper left')
    fig.tight_layout()  
    
    from matplotlib.pyplot import xlim
    xlim(right=250)  
    xlim(left=0)  
    plt.show()
    
def plot_IoU_Stats( datasetName, iou_dict, path_save):
    
    legend_size = 6
    y_axes_size = 12
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    Normal_IoU_Metric = iou_dict['Normal_IoU_Metric']
    x1_vals = list(range(Normal_IoU_Metric.shape[0]))
    color_x1 = 'k'
    ax1.plot(x1_vals, Normal_IoU_Metric, color=color_x1)
    ax1.set_ylabel('Normal Loss', color=color_x1, fontsize = y_axes_size)
    
    color_x2 = 'tab:blue'
    
    IoU_Metric_0_0_1 = iou_dict['0.01']['0.01_IoU']
    IoU_Metric_0_0_5 = iou_dict['0.05']['0.05_IoU']
    IoU_Metric_0_1 = iou_dict['0.1']['0.1_IoU']
    IoU_Metric_0_25 = iou_dict['0.25']['0.25_IoU']
    IoU_Metric_0_5 = iou_dict['0.5']['0.5_IoU']
    IoU_Metric_0_75 = iou_dict['0.75']['0.75_IoU']
    IoU_Metric_1 = iou_dict['1.0']['1.0_IoU']
    
    x2_vals = list(range(IoU_Metric_0_1.shape[0]))
    
    ax2.plot(x2_vals, IoU_Metric_0_0_1, label='IoU-Loss(0.01)')
    ax2.plot(x2_vals, IoU_Metric_0_0_5, label='IoU-Loss(0.05)')
    ax2.plot(x2_vals, IoU_Metric_0_1, label='IoU-Loss(0.1)')
    ax2.plot(x2_vals, IoU_Metric_0_25, label='IoU-Loss(0.25)')
    ax2.plot(x2_vals, IoU_Metric_0_5, label='IoU-Loss(0.5)')
    ax2.plot(x2_vals, IoU_Metric_0_75, label='IoU-Loss(0.75)')
    ax2.plot(x2_vals, IoU_Metric_1, label='IoU-Loss(1)')
    
            
    ax2.set_ylabel('IoU-Loss + LR', color=color_x2, fontsize = y_axes_size)
    ax1.grid()
    
    plt.title( "No-IoU Loss vs IoU Loss : Dataset: " + datasetName)
    
    plt.legend(  fontsize = legend_size)
    fig.tight_layout()
    
    plt.savefig(path_save+"/IoU_Losses_"+datasetName+".pdf")
    plt.show()
    