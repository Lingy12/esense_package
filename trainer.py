from .visualize import plot_instances
from .models import Model
from tensorflow.keras.utils import to_categorical
from .evaluate_tool import get_confusionmatrix, get_derived_mucosal, get_classification_report
from .data_tool import mucous_activity_label_list, non_mucous_activity_label_list

class TrainDriver:
    def __init__(self, **kwarys):
        """Initalize train driver with keyword argument
        """
        
        