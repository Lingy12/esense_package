"""
data_tool.py
=========================
This is the module for create sorted data frame and training data list
"""
import pandas as pd
import time
import numpy as np
from .filters import filter_remove_noise, normalization_minmax, filter_remove_noise_and_gravity
import matplotlib as plt

mucous_activity_label_list = ['[Mucosal]Rub eyes (L)',
                              '[Mucosal]Rub eyes (R)',
                              '[Mucosal]Rub nose bridge',
                              '[Mucosal]Wipe nose',
                              '[Mucosal]Wipe mouth']
non_mucous_activity_label_list = ['[None-M]Touch forehead',
                                  '[None-M]Touch cheek (L)',
                                  '[None-M]Touch cheek (R)',
                                  '[None-M]Touch chin']

activity_label_list = mucous_activity_label_list + non_mucous_activity_label_list

def getDateAndTime(seconds=None):
    return time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime(seconds))

def get_activity_code(activity):
    for i in range(len(activity_label_list)):
        if activity== activity_label_list[i]:
            return i
    return len(activity_label_list)
    
def get_activity_code_arranged(activity):
    if activity == mucous_activity_label_list[0]:
        return 0
    elif activity == mucous_activity_label_list[1]:
        return 1
    elif activity == mucous_activity_label_list[2]:
        return 2
    elif activity == mucous_activity_label_list[3]:
        return 3
    elif activity == mucous_activity_label_list[4]:
        return 4
    elif activity == non_mucous_activity_label_list[0]:
        return 5
    elif activity == non_mucous_activity_label_list[1]:
        return 6
    elif activity == non_mucous_activity_label_list[2]:
        return 7
    elif activity == non_mucous_activity_label_list[3]:
        return 8
    else:
        print(activity,"error")
    
def get_activity_mucous_code(activity):
    if activity in mucous_activity_label_list:
        return 1
    elif activity in non_mucous_activity_label_list:
        return 0
    else:
        print("activity error")

'''
Process the raw data
'''
def data_preprocessing(df_row,data_length,gyro):
    accX = df_row[[str(i) for i in range(data_length)]]
    accX_norm = normalization_minmax(accX)
    if gyro:
        return accX,accX_norm,filter_remove_noise(accX)
    else:
        return accX,accX_norm,filter_remove_noise(accX),filter_remove_noise_and_gravity(accX)
    
def plot_instances(user, activity, session, df, instance_id, data_length, plot=True):
    new_df = df[(df['userid'] == user) & (df['activity'] == activity) & (df['session'] == session) & (df['instance'] == instance_id)]
    if len(new_df) < 6:
        print("data lengh < 6")
        return
       
    instance_df = new_df
    acc_raw_list = []
    gyro_raw_list = []
    
    acc_denoise_list = []
    gyro_denoise_list = []
    acc_denoise_norm_list = []
    gyro_denoise_norm_list = []
    
    acc_denoise_degravity_list = []
    acc_denoise_degravity_norm_list = []
    
    df_row_0 = instance_df.iloc[0,:]
    df_row_1 = instance_df.iloc[1,:]    
    df_row_2 = instance_df.iloc[2,:]
    df_row_3 = instance_df.iloc[3,:]
    df_row_4 = instance_df.iloc[4,:]
    df_row_5 = instance_df.iloc[5,:]
    touching_point = (instance_df["touching point"].iloc[0])
    leaving_point = (instance_df["leaving point"].iloc[0])
    print(touching_point,leaving_point)
    print(instance_df["source"].iloc[0])
    
    if df_row_0["axis"]=="Ax": 
        raw, raw_norm, raw_denoise, raw_denoise_degravity = data_preprocessing(df_row_0,data_length,False)
        acc_raw_list.append(raw)
        acc_denoise_list.append(raw_denoise)
        acc_denoise_norm_list.append(normalization_minmax(raw_denoise))
        acc_denoise_degravity_list.append(raw_denoise_degravity)
        acc_denoise_degravity_norm_list.append(normalization_minmax(raw_denoise_degravity))

    if df_row_1["axis"]=="Ay": 
        raw, raw_norm, raw_denoise, raw_denoise_degravity = data_preprocessing(df_row_1,data_length,False)
        acc_raw_list.append(raw)
        acc_denoise_list.append(raw_denoise)
        acc_denoise_norm_list.append(normalization_minmax(raw_denoise))
        acc_denoise_degravity_list.append(raw_denoise_degravity)
        acc_denoise_degravity_norm_list.append(normalization_minmax(raw_denoise_degravity))
        
    if df_row_2["axis"]=="Az": 
        raw, raw_norm, raw_denoise, raw_denoise_degravity = data_preprocessing(df_row_2,data_length,False)
        acc_raw_list.append(raw)
        acc_denoise_list.append(raw_denoise)
        acc_denoise_norm_list.append(normalization_minmax(raw_denoise))
        acc_denoise_degravity_list.append(raw_denoise_degravity)
        acc_denoise_degravity_norm_list.append(normalization_minmax(raw_denoise_degravity))
        
    if df_row_3["axis"]=="Gx":
        raw, raw_norm, raw_denoise = data_preprocessing(df_row_3,data_length,True)
        gyro_raw_list.append(raw)
        gyro_denoise_list.append(raw_denoise)
        gyro_denoise_norm_list.append(normalization_minmax(raw_denoise))
        
    if df_row_4["axis"]=="Gy": 
        raw, raw_norm, raw_denoise = data_preprocessing(df_row_4,data_length,True)
        gyro_raw_list.append(raw)
        gyro_denoise_list.append(raw_denoise)
        gyro_denoise_norm_list.append(normalization_minmax(raw_denoise))
        
    if df_row_5["axis"]=="Gz": 
        raw, raw_norm, raw_denoise = data_preprocessing(df_row_5,data_length,True)
        gyro_raw_list.append(raw)
        gyro_denoise_list.append(raw_denoise)
        gyro_denoise_norm_list.append(normalization_minmax(raw_denoise))
    
    if plot:
        x_tick = [i/100 for i in range(data_length)]
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(f'User: {user}, Activity: {activity}, Session: {session}, Instance: {instance_id}', fontsize=12)
        plt.subplot(2,3,1)
        plt.plot(x_tick,np.array(acc_raw_list).T.tolist())
        plt.legend(['Ax','Ay','Az'])
        plt.subplot(2,3,2)
        plt.plot(x_tick,np.array(acc_denoise_list).T.tolist())
        plt.legend(['Ax_denoise','Ay_denoise','Az_denoise'])
        plt.subplot(2,3,3)
        plt.plot(x_tick,np.array(acc_denoise_degravity_list).T.tolist())
        plt.legend(['Ax_denoise_degravity','Ay_denoise_degravity','Az_denoise_degravity'])
        plt.subplot(2,3,4)
        plt.plot(x_tick,np.array(gyro_raw_list).T.tolist())
        plt.legend(['Gx','Gy','Gz'])
        plt.subplot(2,3,5)
        plt.plot(x_tick,np.array(gyro_denoise_list).T.tolist())
        plt.axvspan(touching_point/100, (touching_point+2)/100, facecolor='r', alpha=1)
        plt.axvspan(leaving_point/100, (leaving_point+2)/100, facecolor='b', alpha=1)
        plt.legend(['Gx_denoise','Gy_denoise','Gz_denoise'])
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_tick,np.array(gyro_denoise_list).T.tolist())
        plt.legend(['Gx','Gy','Gz'])
        plt.ylabel("degrees per second")
        plt.xlabel("time (second)")
        plt.show()
    return acc_raw_list, acc_denoise_norm_list, acc_denoise_degravity_norm_list, gyro_raw_list, gyro_denoise_norm_list

class Dataset:
    """
    The imu file should be a .csv file and label file should be .xlss file
    """
    def __init__(self, imu_file_name, label_file_name):
        self.df_facetouch_imu = pd.read_csv(imu_file_name, index_col=[0])
        # Remove Cover Mouth
        # re-arrange activities
        self.df_facetouch_imu = self.df_facetouch_imu.drop(self.df_facetouch_imu[self.df_facetouch_imu.activity == "Cover mouth"].index)
        # df_facetouch_imu_sorted = df_facetouch_imu_sorted.drop(df_facetouch_imu_sorted[df_facetouch_imu_sorted.activity == "Touch nose"].index)
        self.df_facetouch_imu = self.df_facetouch_imu.reset_index(drop=True)
        utc_time = self.df_facetouch_imu['utc']
        utc_time_str = [getDateAndTime(int(utc/1000+8*3600)) for utc in utc_time] ## +8*3600 means change to UTC+8
        self.df_facetouch_imu["timestamp"] = utc_time_str
        self.df_facetouching_point_labeling = pd.read_excel(label_file_name)
        self.df_facetouch_imu_sorted = self.df_facetouch_imu[self.df_facetouch_imu['session']>-1]
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.sort_values(by=['userid','activity','session','timestamp','axis']).reset_index(drop=True)
        print("IMU data sorted")
        
        df_facetouching_point_labeling_6x = pd.DataFrame(np.repeat(self.df_facetouching_point_labeling.values,6,axis=0))
        df_facetouching_point_labeling_6x.columns = self.df_facetouching_point_labeling.columns
        self.df_facetouch_imu_sorted['touching point'] = df_facetouching_point_labeling_6x['touching point']+100
        self.df_facetouch_imu_sorted['leaving point'] = df_facetouching_point_labeling_6x['leaving point']+100
        self.df_facetouch_imu_sorted['source'] = df_facetouching_point_labeling_6x['source']
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Rub near eyes (under eyes, eyebrows)_Left","[Mucosal]Rub eyes (L)")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Rub near eyes (under eyes, eyebrows)_Right","[Mucosal]Rub eyes (R)")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch nose bridge","[Mucosal]Rub nose bridge")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch nose","[Mucosal]Wipe nose")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Wipe mouth or lips","[Mucosal]Wipe mouth")

        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch forehead or hair","[None-M]Touch forehead")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch cheek_Left","[None-M]Touch cheek (L)")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch cheek_Right","[None-M]Touch cheek (R)")
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted.replace("Touch chin","[None-M]Touch chin")
        print("Label replaced")

        
    def check_original_sorted_df(self):
        print('Below should show equal number: ')
        print(len(self.df_facetouch_imu))
        print(len(self.df_facetouch_imu_sorted))
        if len(self.df_facetouch_imu)==len(self.df_facetouch_imu_sorted): print("session right")

        
    def check_label_length(self):
        print(len(self.df_facetouching_point_labeling))
    
    def print_dataset_statistic(self):
        print("#Users",len(self.df_facetouch_imu_sorted["userid"].unique()))
        print("#Activity",len(self.df_facetouch_imu_sorted["activity"].unique()))
        print("Activity",(self.df_facetouch_imu_sorted["activity"].unique()))
        print("#Instances",len(self.df_facetouch_imu_sorted)/6)
        for activity in self.df_facetouch_imu_sorted["activity"].unique():
            print("#Instance",activity,len(self.df_facetouch_imu_sorted[self.df_facetouch_imu_sorted["activity"]==activity])/6)
    
    def get_sorted_df(self):
        return self.df_facetouch_imu_sorted
    
class DataGenerator:
    '''
    Initialize with a raw data frame
    '''
    def __init__(self, df):
        self.df = df
        self.imu_instance_list = []
        self.imu_instance_normalized_list = []
        self.label_list = []
        self.label_mucous_list = []
        self.user_list = []
        self.session_list = []
        self.session_instance_list = []
        self.window_id_list = []
        self.imu_instance_following_list = []
        self.time_to_touch_list = []
    
    '''
    Generate the data
    Note: Pretouch only must be true if predicting t
    Set data following length to 0 for non-forcasting task
    
    data_length: Length of the input data
    step_size: step size for sliding window
    data_following_length: Forcasted target length
    session_exclude: exclude 1 to exclude unguided, set to 0 if want to include unguided
    source_include: include the source of label
    user_exclude: leave a user out
    for_test: for leave one-user-out test or not
    user_only: include the user for test
    pre_touch_only: only extracting pre_touch window or not
    '''
    def generate_data(self, data_length,step_size, window_num, data_following_length, 
                      session_exclude = 1, source_include = ['video'], 
                      user_exclude = -1, for_test = False, user_only = -1, pre_touch_only = True):
        assert for_test == False or user_only > 0 # Ensure the for_test triggered correctly

        for i in range(int(len(self.df) / 6)):
            df_row_0 = self.df.iloc[i * 6, :]
            
            # Controlling for target
            if df_row_0['session'] == session_exclude:
                continue
            if df_row_0['source'] not in source_include:
                continue
            if df_row_0['userid'] == user_exclude and not for_test :
                continue
            if for_test and user_only != df_row_0['userid']:
                continue
            
            # Get touching point
            touch_touching_point = int(df_row_0['touching point'])
            if touch_touching_point<data_length or touch_touching_point>400:
                print("touching point label error")
                continue
            
            # Generate windows
            for j in range(window_num):
                imu_list = []
                imu_normalized_list = []
                imu_following_list = []
                
                if pre_touch_only:
                    data_start = touch_touching_point - j * step_size - data_length
                    data_end = touch_touching_point - j * step_size + data_following_length
                else:
                    data_start = 20 + j * step_size
                    data_end = 20 + j * step_size + data_length
                    
                if data_start < 20:
                    print("start data point out of boundry!", i, j)
                    continue
                if data_end >= 400:
                    print("end data point out of boundry!", i, j)
                    
                if df_row_0["axis"]=="Ax": 
                    accX = df_row_0[[str(i) for i in range(data_start,data_end)]]
                    accX_f = filter_remove_noise_and_gravity(accX)
                    imu_list.append(accX_f[:data_length])
                    imu_following_list.append(accX_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(accX_f))
                df_row_1 = self.df.iloc[i*6+1,:]
                if df_row_1["axis"]=="Ay": 
                    accY = df_row_1[[str(i) for i in range(data_start,data_end)]]
                    accY_f = filter_remove_noise_and_gravity(accY)
                    imu_list.append(accY_f[:data_length])
                    imu_following_list.append(accY_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(accY_f))
                df_row_2 = self.df.iloc[i*6+2,:]
                if df_row_2["axis"]=="Az": 
                    accZ = df_row_2[[str(i) for i in range(data_start,data_end)]]
                    accZ_f = filter_remove_noise_and_gravity(accZ)
                    imu_list.append(accZ_f[:data_length])
                    imu_following_list.append(accZ_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(accZ_f))
                df_row_3 = self.df.iloc[i*6+3,:]
                if df_row_3["axis"]=="Gx":
                    gyro_deg = df_row_3[[str(i) for i in range(data_start,data_end)]]
            #         gyro_rad = [math.radians(gyro)/math.pi for gyro in gyro_deg]
                    gyro_deg_f = filter_remove_noise(gyro_deg)
                    imu_list.append(gyro_deg_f[:data_length])
                    imu_following_list.append(gyro_deg_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(gyro_deg_f))
                df_row_4 = self.df.iloc[i*6+4,:]
                if df_row_4["axis"]=="Gy": 
                    gyro_deg = df_row_4[[str(i) for i in range(data_start,data_end)]]
            #         gyro_rad = [math.radians(gyro)/math.pi for gyro in gyro_deg]
                    gyro_deg_f = filter_remove_noise(gyro_deg)
                    imu_list.append(gyro_deg_f[:data_length])
                    imu_following_list.append(gyro_deg_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(gyro_deg_f))
                df_row_5 = self.df.iloc[i*6+5,:]
                if df_row_5["axis"]=="Gz": 
                    gyro_deg = df_row_5[[str(i) for i in range(data_start,data_end)]]
            #         gyro_rad = [math.radians(gyro)/math.pi for gyro in gyro_deg]
                    gyro_deg_f = filter_remove_noise(gyro_deg)
                    imu_list.append(gyro_deg_f[:data_length])
                    imu_following_list.append(gyro_deg_f[data_length:])
        #             imu_normalized_list.append(normalization_minmax(gyro_deg_f))

                if len(imu_list)!=6:
                    print(i)
                else:
                    add_gravity = False
                    if add_gravity:
                        imu_list.append(accX_f) ## add acc data with gravity
                        imu_list.append(accY_f)
                        imu_list.append(accZ_f)
                        imu_normalized_list.append(normalization_minmax(accX_f)) ## add acc data with gravity
                        imu_normalized_list.append(normalization_minmax(accY_f))
                        imu_normalized_list.append(normalization_minmax(accZ_f))           

                    self.imu_instance_list.append(np.array(imu_list).T.tolist())
        #             imu_instance_normalized_list.append(np.array(imu_normalized_list).T.tolist())
                    self.label_mucous_list.append([get_activity_mucous_code(df_row_0['activity'])])
                    self.label_list.append([get_activity_code_arranged(df_row_0['activity'])])
                
                    self.imu_instance_following_list.append(np.array(imu_following_list).T.tolist())

                    self.user_list.append(df_row_0['userid'])
                    self.session_list.append(df_row_0['session'])
                    self.session_instance_list.append(df_row_0['instance'])
                    self.window_id_list.append(j)
                    self.time_to_touch_list.append(float(touch_touching_point - data_end) / 100) # TODO: Confirm this              
    
    '''
    Get the data and target 
    '''            
    def get_list_for_forcasting(self):
        assert len(self.imu_instance_list) != 0 and len(self.imu_instance_following_list) != 0
        return self.imu_instance_list, self.imu_instance_following_list
    
    def get_list_for_classification(self):
        assert len(self.imu_instance_list) != 0 and len(self.label_list) != 0 and len(self.label_list) != 0
        return self.imu_instance_list, self.label_list, self.label_mucous_list
    
    def get_list_for_time_prediction(self):
        assert len(self.imu_instance_list) != 0 and len(self.time_to_touch_list) != 0
        return self.imu_instance_list, self.time_to_touch_list
    
    # Reset all list
    def reset(self):
        self.imu_instance_list = []
        self.imu_instance_normalized_list = []
        self.label_list = []
        self.label_mucous_list = []
        self.user_list = []
        self.session_list = []
        self.session_instance_list = []
        self.window_id_list = []
        self.imu_instance_following_list = []
        self.time_to_touch_list = []