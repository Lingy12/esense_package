from .visualize import plot_instances
from .models import Model
from tensorflow.keras.utils import to_categorical
from .evaluate_tool import get_confusionmatrix, get_derived_mucosal, get_classification_report
from .data_tool import mucous_activity_label_list, non_mucous_activity_label_list
import tensorflow as tf

class TrainHelper:
    """Train helper helps the to train the given model with different requirements
    """
    def __init__(self, model:tf.keras.Model, model_name:str, log:bool = False):
        """Initialize the train helper
        """
        self.model_name = model_name
        self.logging_path = f'./my_checkpoint/{model_name}'
        self.logging_dir = os.path.dirname(self.logging_path)
        self.log = log
        self.model = model
    
    def train_model(self, x, y, criteria:str = 'val_accuracy', **kwargs):
        """Fit the model.

        Args:
            x (array like): train input.
            y (array like): train target.
            criteria (str, optional): Criteria to save the model. Defaults to 'val_accuracy'.
        """
        assert len(x) == len(y)
        if self.log == True:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.logging_path,
                                                 save_weights_only=True,
                                                 monitor = criteria,
                                                 verbose=kwargs['verbose'], save_best_only=True)
            if kwargs.__contains__('callback'):
                kwargs['callbacks'].append(cp_callback)
            else:
                kwargs['callbacks'] = [cp_callback]
        
        return self.model.fit(x, y, **kwargs)
    
    def evaluate_model(self, x, y):
        return self.model.evaluate(x, y)