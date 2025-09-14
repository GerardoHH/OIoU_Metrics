import tensorflow as tf
 
def iou_nd(boxes1, boxes2, D):
    #Calcula el IoU en N dimensiones entre dos lotes de hipercubos
    inter_min = tf.maximum(boxes1[..., :D], boxes2[..., :D])  
    inter_max = tf.minimum(boxes1[..., D:], boxes2[..., D:])  

    inter_vol = tf.reduce_prod(tf.maximum(inter_max - inter_min, 0), axis=-1)
    
    vol1 = tf.reduce_prod(boxes1[..., D:] - boxes1[..., :D], axis=-1)
    vol2 = tf.reduce_prod(boxes2[..., D:] - boxes2[..., :D], axis=-1)
    
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / tf.maximum(union_vol, 1e-6)

@tf.function(jit_compile=True)  
def IoU_One_vs_all_nd_Permutacion(w_min_vect, w_max_vect, D):
    
    num_elements = tf.shape(w_min_vect)[0]
    
    boxes = tf.concat([w_min_vect, w_max_vect], axis=1)
    
    boxes_exp1 = tf.expand_dims(boxes, axis=1)  
    boxes_exp2 = tf.expand_dims(boxes, axis=0)  
    
    iou_matrix = iou_nd(tf.broadcast_to(boxes_exp1, [num_elements, num_elements, 2*D]), 
                        tf.broadcast_to(boxes_exp2, [num_elements, num_elements, 2*D]),
                        D)

    iou_sum = tf.reduce_sum(iou_matrix, axis=1) - tf.linalg.tensor_diag_part(iou_matrix)
    
    return tf.reduce_sum(iou_sum)
    
@tf.function(jit_compile=True)  
def IoU_unique_pairs_nd_Combinacion(w_min_vect, w_max_vect, D):

    num_elements = tf.shape(w_min_vect)[0]
    boxes = tf.concat([w_min_vect, w_max_vect], axis=1)

    boxes_exp1 = tf.expand_dims(boxes, axis=1)  
    boxes_exp2 = tf.expand_dims(boxes, axis=0)

    iou_matrix = iou_nd(
        tf.broadcast_to(boxes_exp1, [num_elements, num_elements, 2 * D]),
        tf.broadcast_to(boxes_exp2, [num_elements, num_elements, 2 * D]),
        D
    )

    i = tf.range(num_elements)
    j = tf.range(num_elements)
    ii, jj = tf.meshgrid(i, j, indexing='ij')
    mask_upper = ii < jj  

    ious = tf.boolean_mask(iou_matrix, mask_upper)
    iou_avg = tf.reduce_mean(ious) *100.0

    return iou_avg