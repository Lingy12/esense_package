import tensorflow as tf
import os

class TrainHelper:
    """Train helper helps the to train the given model with different requirements.
    """
    def __init__(self, model:tf.keras.Model, model_name:str, log:bool = False):
        """Initialize the helper by passing a model and it's name. 

        Args:
            model (tf.keras.Model): Compiled tensorflow model to train.
            model_name (str): Name of the model.
            log (bool, optional): Whether save the checkpoint or not. Defaults to False.
        """
        self.model_name = model_name
        self.logging_path = f'./my_checkpoint/{model_name}'
        self.logging_dir = os.path.dirname(self.logging_path)
        self.log = log
        self.model = model
    
    def train_model(self, x, y, criteria:str = 'val_loss', **kwargs):
        """Fit the model with keyward arguments as tf.keras.Model.fit.

        Args:
            x (array like): train input.
            y (array like): train target.
            criteria (str, optional): Criteria to save the model. Defaults to 'val_accuracy'.
        """
        if self.log == True:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.logging_path,
                                                 save_weights_only=True,
                                                 monitor = criteria,
                                                 verbose=kwargs['verbose'], save_best_only=True)
            if kwargs.__contains__('callbacks'):
                kwargs['callbacks'].append(cp_callback)
            else:
                kwargs['callbacks'] = [cp_callback]
        
        return self.model.fit(x, y, **kwargs)
    
    def evaluate_model(self, x, y, **kwargs):
        """Evaluate the model with test data.

        Args:
            x (array like): test input.
            y (array like): test label.

        Returns:
            _type_: _description_
        """
        self.model.load_weights(self.logging_path)
        return self.model.evaluate(x, y, **kwargs)
    
    def get_best_model(self) -> tf.keras.Model:
        """Load the saved model.

        Returns:
            tf.keras.Model: Trained model
        """
        self.model.load_weights(self.logging_path)
        return self.model