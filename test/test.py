from ..data_tool import Dataset, DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
from tensorflow.keras.utils import to_categorical
import numpy as np
from ..models import Model

dataset = Dataset(imu_file_name = '/data1/esense/IMU_20220408_1.5s.csv', label_file_name = "/data1/esense/facetouch_imu_sorted_for_timing_label_20220217_combine.xlsx")
print(dataset.check_original_sorted_df())
print(dataset.print_dataset_statistic())
print(dataset.check_label_length())
df = dataset.get_sorted_df()
assert df != None

generator = DataGenerator(df)

data_length = 120 ## adjust this
step_size = 5 # window step 0.1s
window_num = 8 # number of sliding windows
generator.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                        data_following_length = int((step_size * window_num)*1.5),
                       user_exclude = 5)

trainX, testX, trainy, testy, trainIDs, testIDs = train_test_split(np.array(imu_instance_list), 
                                                np.array(to_categorical(imu_label_list)), 
                                                np.array(list(range(len(imu_instance_list)))),
                                                test_size=0.2, random_state=0, stratify = imu_label_list)

verbose, epochs, batch_size = 1, 1, 1
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Model('None')
model.init_1d_cnn_model(64, 10,(n_timesteps,n_features), n_outputs)
model.print_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
_, accuracy = model.evaluate(trainX, trainy, batch_size=batch_size, verbose=0)

