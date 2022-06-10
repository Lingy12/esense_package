from filters import filter_remove_noise, filter_remove_noise_and_gravity, normalization_minmax, normalization_standard
import matplotlib.pyplot as plt

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