"""
data_tool.py
================================================================
This is the module for create sorted data frame and training data list
"""
import pandas as pd
import time
import numpy as np
from .filters import filter_remove_noise, normalization_minmax, filter_remove_noise_and_gravity
import matplotlib as plt
from sklearn.model_selection import train_test_split

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

def getDateAndTime(seconds:str=None) -> str:
    """Get the date and time in required format

    Args:
        seconds (str, optional): utc string in any format. Defaults to None.

    Returns:
        date time string in %Y/%m/%d %H:%M:%S.
    """
    return time.strftime("%Y/%m/%d %H:%M:%S", time.gmtime(seconds))

def get_activity_code(activity:str) -> int:
    """Get activity code from activity string.

    Args:
        activity (str): activity name.

    Returns:
        int: code representation for activity.
    """
    for i in range(len(activity_label_list)):
        if activity== activity_label_list[i]:
            return i
    return len(activity_label_list)
    
def get_activity_code_arranged(activity:str) -> int:
    """Arrange activity according mucosal / non-mucosal.

    Args:
        activity (str): activity name.

    Returns:
        int: arranged code representation for activity.
    """
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
        return -1
    
def get_activity_mucous_code(activity:str) -> int:
    """Return mucous/non-mucous label for activity.

    Args:
        activity (str): activity name.

    Returns:
        int: representation code for activity.
    """
    if activity in mucous_activity_label_list:
        return 1
    elif activity in non_mucous_activity_label_list:
        return 0
    else:
        print("activity error")
        return -1

class Dataset:
    """Dataset represent a sorted / processed dataframe.
    """
    def __init__(self, imu_file_name:str, label_file_name:str):
        """Initialize dataset with imu file and label file.

        Args:
            imu_file_name (str): file path to IMU file.
            label_file_name (str): file path to label file.
        """
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
        self.train_df = None
        self.test_df = None
        self.total_instance = len(self.df_facetouch_imu_sorted) / 6 # six axis
        #total 2028 instance

    def get_sorted_df(self) -> pd.DataFrame:
        """Extract the dataframe constructed from the imu file and label file

        Returns:
            pd.DataFrame: result data frame
        """
        return self.df_facetouch_imu_sorted
    
    def get_train_df(self) -> pd.DataFrame:
        """Return the train data frame.

        Returns:
            pd.DataFrame: train data frame.
        """
        return self.train_df

    def get_test_df(self) -> pd.DataFrame:
        """Return the test data frame

        Returns:
            pd.DataFrame: test data frame
        """
        return self.test_df
    
    def deploy_leave_user(self, userid:int):
        """Support for leave one user out

        Args:
            userid (int): user id
        """
        assert userid in self.df_facetouch_imu_sorted['userid'].unique()
        self.train_df = self.df_facetouch_imu_sorted[self.df_facetouch_imu_sorted['userid'] != userid]
        self.test_df = self.df_facetouch_imu_sorted[self.df_facetouch_imu_sorted['userid'] == userid]
    
    def deploy_train_test_split(self, test_size: float):
        """Train test split the data frame

        Args:
            test_size (float): size for test
        """
        
        index_train, index_test = train_test_split(self.df_facetouch_imu_sorted[self.df_facetouch_imu_sorted['axis'] == 'Ax'].index)
        index_train, index_test = list(index_train), list(index_test)
        
        train_size = len(index_train)
        test_size = len(index_test)
        for i in range(1,6):
            for j in range(train_size):
                index_train.append(index_train[j] + i)
            for k in range(test_size):
                index_test.append(index_test[k] + i)
        
        self.train_df = self.df_facetouch_imu_sorted.iloc[index_train].sort_values(by=['userid','activity','session','timestamp','axis']).reset_index(drop=True)
        self.test_df = self.df_facetouch_imu_sorted.iloc[index_test].sort_values(by=['userid','activity','session','timestamp','axis']).reset_index(drop=True)            
    
    def check_original_sorted_df(self):
        """Check whether the dataframe is correct after sorting.
        """
        print('Below should show equal number: ')
        print(len(self.df_facetouch_imu))
        print(len(self.df_facetouch_imu_sorted))
        if len(self.df_facetouch_imu)==len(self.df_facetouch_imu_sorted): print("session right")
    
    def check_label_length(self):
        """Check the length for label file.
        """
        print(len(self.df_facetouching_point_labeling))
    
    def print_dataset_statistic(self):
        """print out the statistic for dataset.
        """
        print("#Users",len(self.df_facetouch_imu_sorted["userid"].unique()))
        print("#Activity",len(self.df_facetouch_imu_sorted["activity"].unique()))
        print("Activity",(self.df_facetouch_imu_sorted["activity"].unique()))
        print("#Instances",len(self.df_facetouch_imu_sorted)/6)
        for activity in self.df_facetouch_imu_sorted["activity"].unique():
            print("#Instance",activity,len(self.df_facetouch_imu_sorted[self.df_facetouch_imu_sorted["activity"]==activity])/6)
    
    def filter_source_session(self, source:list, session: list):
        """Filter the dataframe according to session and source

        Args:
            source (str): source of the data
            session (int): session of the data
        """
        self.df_facetouch_imu_sorted = self.df_facetouch_imu_sorted[(self.df_facetouch_imu_sorted['session'].isin(session)) & self.df_facetouch_imu_sorted['source'].isin(source)].reset_index()
        
class DataGenerator:
    """Generate data for training purpose.
    """
    def __init__(self, df:pd.DataFrame):
        """Initialize data generator with a processed data frame
        
        Args:
            df (pd.DataFrame): raw data in dataframe format
        """
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
    
    data_length: Length of the input data.
    step_size: step size for sliding window.
    data_following_length: Forcasted target length.
    session_exclude: exclude 1 to exclude unguided, set to 0 if want to include unguided.
    source_include: include the source of label.
    user_exclude: leave a user out.
    for_test: for leave one-user-out test or not.
    user_only: include the user for test.
    pre_touch_only: only extracting pre_touch window or not.
    '''
    def generate_data(self, data_length:int,step_size:int, window_num:int, data_following_length:int, pre_touch_only: bool = False):
        """Produce data for different purpose and stored in the object

        Args:
            data_length (int): length of a sliding window
            step_size (int): step size of the generating process
            window_num (int): target number of window
            data_following_length (int): target forcasted signal length for a window

        """
        # assert for_test == False or user_only > 0 # Ensure the for_test triggered correctly
        for i in range(int(len(self.df) / 6)):
            df_row_0 = self.df.iloc[i * 6, :]
            
            # # Controlling for target
            # if df_row_0['session'] == session_exclude:
            #     continue
            # if df_row_0['source'] not in source_include:
            #     continue
            # if df_row_0['userid'] == user_exclude and not for_test :
            #     continue
            # if for_test and user_only != df_row_0['userid']:
            #     continue
            
            # Get touching point
            touch_touching_point = int(df_row_0['touching point'])
            if touch_touching_point>400:
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
                if data_end > 400:
                    print("end data point out of boundry!", i, j)
                    continue
                
                # Ensure the overlapping of touching point
                if data_start > touch_touching_point or data_end < touch_touching_point:
                    # print('Skipping non-overlapping')
                    continue
                
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
    def get_list_for_forcasting(self) -> tuple:
        """Get training data for time series forcasting task.

        Returns:
            tuple: raw series, forcasted target seris.
        """
        assert len(self.imu_instance_list) != 0 and len(self.imu_instance_following_list) != 0
        return self.imu_instance_list, self.imu_instance_following_list
    
    def get_list_for_classification(self) -> tuple:
        """Get training data for classification task.

        Returns:
            tuple: raw series and labels (raw label and mucous label).
        """
        assert len(self.imu_instance_list) != 0 and len(self.label_list) != 0 and len(self.label_list) != 0
        return self.imu_instance_list, self.label_list, self.label_mucous_list
    
    def get_list_for_time_prediction(self) -> tuple:
        """Get training data for touching time prediction.

        Returns:
            tuple: raw series and time to touch label.
        """
        assert len(self.imu_instance_list) != 0 and len(self.time_to_touch_list) != 0
        return self.imu_instance_list, self.time_to_touch_list
    
    # Reset all list
    def reset(self):
        """reset the generated data.
        """
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