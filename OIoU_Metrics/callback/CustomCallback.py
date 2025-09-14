import tensorflow as tf
import ModelUtils as mU 
import iou.Tensor_IoU as tf_IoU

class CustomCallback(tf.keras.callbacks.Callback):
         
    def on_epoch_end(self, epoch, logs=None):        
        if (epoch == 0):
            mU.epoch_callback_weigths = []
            
        iou_tf_combination = tf_IoU.IoU_unique_pairs_nd_Combinacion(
                                           (self.model.get_layer(index=0).get_weights()[0]),
                                           (self.model.get_layer(index=0).get_weights()[1]),
                                           tf.shape(self.model.get_layer(index=0).get_weights()[0])[1] ) 


        mU.epoch_callback_weigths.append(iou_tf_combination)
