import tensorflow as tf
def activity_musocal_loss(hp_mucosal):
    def loss(y_true, y_pred):
        # differentiable argmax
        def softargmax(x, beta=1e10): # ref: https://stackoverflow.com/a/54294985
            x = tf.convert_to_tensor(x)
            x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
            return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

        # note how y_pred is one-hot encoded
        # assume that top-k elements in the vector are mucosal, and remainder are non-mucosal
        mucosal_activities_mask = tf.constant([1.,1.,1.,1.,1.,0.,0.,0.,0.]) # same length as one-hot y_pred
        index_pred = softargmax(y_pred)
        y_mucosal_true = tf.math.multiply(y_true, mucosal_activities_mask)
        y_mucosal_pred = tf.math.multiply(y_pred, mucosal_activities_mask)
        #         tf.print(y_true.shape)
        #         tf.print(y_mucosal_true.shape)
        #         tf.print(mucosal_activities_mask.shape)
        #         tf.print(y_true.shape[0])
        

        non_mucosal_activities_mask = tf.constant([0.,0.,0.,0.,0.,1.,1.,1.,1.]) # same length as one-hot y_pred
        y_non_mucosal_true = tf.math.multiply(y_true, non_mucosal_activities_mask)
        y_non_mucosal_pred = tf.math.multiply(y_pred, non_mucosal_activities_mask)

        cce = tf.keras.losses.CategoricalCrossentropy()
        activity_loss = cce(y_true, y_pred)

        total_loss = activity_loss + hp_mucosal * tf.where(index_pred < 5, cce(y_mucosal_true, y_mucosal_pred), 
                                cce(y_non_mucosal_true, y_non_mucosal_pred))

        return total_loss
    return loss

def activity_musocal_loss2(hp_mucosal):
    def loss(y_true, y_pred):
        # differentiable argmax
        def softargmax(x, beta=1e10): # ref: https://stackoverflow.com/a/54294985
            x = tf.convert_to_tensor(x)
            x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
            return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

        all_one = tf.ones_like(y_true)
        
        # note how y_pred is one-hot encoded
        # assume that top-k elements in the vector are mucosal, and remainder are non-mucosal
        mucosal_activities_mask = tf.constant([1.,1.,1.,1.,1.,0.,0.,0.,0.]) # same length as one-hot y_pred
        index_pred = softargmax(y_pred)
        #         y_mucosal_true = tf.math.multiply(y_true, mucosal_activities_mask)
        y_mucosal_pred = tf.math.multiply(y_pred, mucosal_activities_mask)
        mucosal_activities_mask_expanded = tf.math.multiply(all_one, mucosal_activities_mask/5.0)
        
        non_mucosal_activities_mask = tf.constant([0.,0.,0.,0.,0.,1.,1.,1.,1.]) # same length as one-hot y_pred
        #         y_non_mucosal_true = tf.math.multiply(y_true, non_mucosal_activities_mask)
        y_non_mucosal_pred = tf.math.multiply(y_pred, non_mucosal_activities_mask)
        non_mucosal_activities_mask_expanded = tf.math.multiply(all_one, non_mucosal_activities_mask/4.0)

        cce = tf.keras.losses.CategoricalCrossentropy()
        activity_loss = cce(y_true, y_pred)
        
        ## note here I use the mask as label, that I reward the predictions within the mucosal/non-mucosal subgroup
        total_loss = activity_loss + hp_mucosal * tf.where(index_pred < 5, 
                                cce(mucosal_activities_mask_expanded, y_mucosal_pred), 
                                cce(non_mucosal_activities_mask_expanded, y_non_mucosal_pred))

        return total_loss
    return loss

def activity_musocal_loss_brian(hp_mucosal):
    def loss(y_true, y_pred):
        mucosal_activities_mask = tf.constant([1.,1.,1.,1.,1.,0.,0.,0.,0.]) # same length as one-hot y_pred
        mucosal_activities_mask = tf.stack([mucosal_activities_mask, 1 - mucosal_activities_mask],axis=1) # make into Nx2 matrix with 1 vector for mucosal, other for non-mucosal
        y_mucosal_true = tf.matmul(y_true, mucosal_activities_mask) # 1-hot encoded vector format: [sum_probs of mucosal, sum_probs of non-mucosal]
        y_mucosal_pred = tf.matmul(y_pred, mucosal_activities_mask)

        cce = tf.keras.losses.CategoricalCrossentropy()
        activity_loss = cce(y_true, y_pred)
        mucosal_loss = cce(y_mucosal_true, y_mucosal_pred)
        
        total_loss = activity_loss + hp_mucosal * mucosal_loss
        return total_loss

    return loss

def activity_musocal_loss_yunlong(hp_mucosal):
    def loss(y_true, y_pred):
        all_one = tf.ones_like(y_true)
        # first 5 elements in the vector are mucosal, and rest 4 are non-mucosal
        mucosal_activities_mask = tf.constant([1.,1.,1.,1.,1.,0.,0.,0.,0.]) 
        y_mucosal_pred = tf.math.multiply(y_pred, mucosal_activities_mask) ## [batch_size x 9]
        y_mucosal_pred_sum = tf.reduce_sum(y_mucosal_pred,axis=-1,keepdims=True) ## [batch_size x 1]
        mucosal_activities_mask_expanded = tf.math.multiply(all_one, mucosal_activities_mask)
        y_true_mucosal_expanded = tf.math.multiply(mucosal_activities_mask_expanded, y_true)
        mucosal_true = tf.reduce_sum(y_true_mucosal_expanded,axis=-1,keepdims=True) ## [batch_size x 1]
        
        non_mucosal_activities_mask = 1.-mucosal_activities_mask
        y_non_mucosal_pred = tf.math.multiply(y_pred, non_mucosal_activities_mask)
        non_mucosal_activities_mask_expanded = tf.math.multiply(all_one, non_mucosal_activities_mask)
        y_non_mucosal_pred_sum = tf.reduce_sum(y_non_mucosal_pred,axis=-1,keepdims=True)
        y_true_non_ucosal_expanded = tf.math.multiply(non_mucosal_activities_mask_expanded, y_true)
        non_mucosal_true = tf.reduce_sum(y_true_non_ucosal_expanded,axis=-1,keepdims=True)
        
        y_mucosal_non_mucosal_pred = tf.concat([y_mucosal_pred_sum,y_non_mucosal_pred_sum],-1) ## [batch_size x 2]
        y_mucosal_non_mucosal_true = tf.concat([mucosal_true, non_mucosal_true],-1) ## [batch_size x 2]
        
        cce = tf.keras.losses.CategoricalCrossentropy()
        activity_loss = cce(y_true, y_pred)
        mucosal_loss = cce(y_mucosal_non_mucosal_true, y_mucosal_non_mucosal_pred)
        total_loss = activity_loss + hp_mucosal * mucosal_loss
        return total_loss
    return loss