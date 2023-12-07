from pandas import read_csv, DataFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from RNNmodel_example import create_RNN

### Reading and Split the data
def get_train_test(url, columns, split_percent):
    df = read_csv(url, engine='python')  
    selected_columns = columns
    df_selected = df[selected_columns]
    data = np.array(df_selected.values.astype('float32'))
    n = len(data)
    # Point for splitting data into train and test
    split = int(n * split_percent)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data, data

CapacityCsv = 'D:\Jeff\GP\Capacity_raw.csv'
train_data, test_data, data = get_train_test(CapacityCsv, ['Time', 'Current', 'Cell_Voltage'], 0.8)

# Separate columns for DataFrame
time_column = train_data[:, 0]  # Assuming Time is the first column
current_column = train_data[:, 1]  # Assuming Current is the second column
voltage_column = train_data[:, 2]  # Assuming Cell_Voltage is the third column

### Reshape the data for Keras
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    
    # Prepare X
    rows_x = len(Y)
    num_features = dat.shape[1]  # Assuming dat is a 2D array with multiple features
    X = np.zeros((rows_x, time_steps, num_features))
    
    for i in range(rows_x):
        start_idx = i * time_steps
        end_idx = start_idx + time_steps
        X[i] = dat[start_idx:end_idx, :]
    
    return X, Y

time_steps=10
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

### try to know what do we have inside the trainX (Unnecessary)
import matplotlib.pyplot as plt

# Plot the first few samples in trainX
feature_names = ['Current', 'Cell_Voltage']

# Plot the first few samples in trainX with time on the x-axis
for i in range(min(10, trainX.shape[0])):  # Plot at most 5 samples
    time_values = trainX[i, :, 0]  # Assuming time is the first column
    for j in range(1, trainX.shape[2]):  # Start from the second column (skip time)
        plt.plot(time_values, trainX[i, :, j], label=f'{feature_names[j-1]}')  # Adjust index to skip time
    plt.title(f'Sample {i+1}')
    plt.xlabel('Time')
    plt.legend()
    plt.show()
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot all samples in trainX on the same chart
for i in range(min(5, trainX.shape[0])):  # Plot at most 5 samples
    time_values = trainX[i, :, 0]  # Assuming time is the first column
    for j in range(1, trainX.shape[2]):  # Start from the second column (skip time)
        plt.plot(time_values, trainX[i, :, j], label=f'{feature_names[j-1]} (Sample {i+1})')  # Adjust index to skip time

plt.title('Combined Plot of Multiple Samples')
plt.xlabel('Time')
plt.legend()
plt.show()