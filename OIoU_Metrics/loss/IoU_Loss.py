import tensorflow as tf
import keras.backend as K
from iou import Tensor_IoU as tf_IoU

def iou_loss(model, num_classes, lr =1.):
    
    def loss(y_true, y_pred):

        iou_tf = tf_IoU.IoU_unique_pairs_nd_Combinacion( 
                    (model.layers[0].weights[0]),
                    (model.layers[0].weights[1]),
                     tf.shape(model.layers[0].weights[0])[1])

        iou_tf = iou_tf*lr
        
        y_true = K.cast(y_true, dtype='float32')
        
        if num_classes > 2:
            classification_loss = K.categorical_crossentropy(y_true, y_pred)
        else:
            classification_loss = K.binary_crossentropy(y_true, y_pred)
        
        total_loss = K.mean(classification_loss) + iou_tf
        
        return total_loss

    return loss  