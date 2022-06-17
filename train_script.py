# Data creation
verbose, epochs, batch_size = 1, 20, 32
dataset = Dataset(imu_file_name = '/data1/esense/IMU_20220408_1.5s.csv', label_file_name = "/data1/esense/facetouch_imu_sorted_for_timing_label_20220217_combine.xlsx")
dataset.filter_source_session(source=['video'], session=[2, 3])
dataset.deploy_train_test_split(0.2)
train_df = dataset.get_train_df()
test_df = dataset.get_test_df()
train_dg = DataGenerator(train_df)
test_dg = DataGenerator(test_df)

train_dg.reset()
test_dg.reset()

# TODO: Change overlapping criteria if data length changes
data_length = 120 ## adjust this
step_size = 5 # window step 0.1s
window_num = 50 # number of sliding windows
train_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                        data_following_length = int((step_size * window_num)*1.5))
test_dg.generate_data(data_length = data_length, step_size = step_size, window_num = window_num, 
                        data_following_length = int((step_size * window_num)*1.5))

trainX, trainy, trainy_bi = train_dg.get_list_for_classification()
testX, testy, testy_bi = test_dg.get_list_for_classification()

testX, testy, testy_bi = np.array(testX), to_categorical(np.array(testy)), to_categorical(np.array(testy_bi))
trainX, trainy, trainy_bi = np.array(trainX), to_categorical(np.array(trainy)), to_categorical(np.array(trainy_bi))

print(f'Training instance: {len(trainX)}')
print(f'Test instance: {len(testX)}')

n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

model = Model('None', log=True)
model.init_1d_cnn_model_regularized(64, 3, (n_timesteps,n_features), n_outputs, feature_num = 100)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.print_model()
hist = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, 
          verbose=verbose, validation_data=(testX, testy))
model.get_best_model()
_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)

y_pred = model.predict(testX)

get_confusionmatrix(y_pred, testy, mucous_activity_label_list + non_mucous_activity_label_list, 'CM')

get_derived_mucosal(y_pred, testy, 'CM [Binary]')

y_pred_mucosal, y_true_mucosal = derive_mucosal(y_pred, testy)
print(get_classification_report(y_pred_mucosal, y_true_mucosal))