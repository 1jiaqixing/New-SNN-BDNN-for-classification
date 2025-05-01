import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
def process_and_split_data(zip_file_path, target_length=37750, timesteps=10):
    new_data = []
    labels = []  # 用于存储每个样本对应的标签
    # 均值化后的目标长度
    new_length = target_length // timesteps
    a = 0

    # 遍历标签 01 到 12
    for label in range(1, 13):
        # 格式化标签，使其为两位数（如 01, 02, ..., 12）
        label_str = f"{label:02d}"
        # 构建 pkl 文件所在的相对路径
        pkl_folder_path = f'pickles_30/texture_{label_str}/full_imu'

        try:
            # 打开压缩文件
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                # 获取该标签文件夹下的所有 pkl 文件
                pkl_files = [name for name in zf.namelist() if name.startswith(pkl_folder_path) and name.endswith('.pkl')]

                # 遍历该标签下的所有 pkl 文件
                for pkl_file_name in pkl_files:
                    a = a + 1
                    # 从压缩文件中读取 pkl 文件的二进制内容
                    pkl_binary = zf.read(pkl_file_name)
                    pkl_file = io.BytesIO(pkl_binary)
                    data = pd.read_pickle(pkl_file)

                    np_array = np.array(data)

                    # 调整数组长度
                    if np_array.size > 0:
                        if np_array.ndim == 1:
                            current_length = len(np_array)
                        else:
                            current_length = np_array.shape[0]

                        if current_length > target_length:
                            # 截断多的部分
                            np_array = np_array[:target_length]
                        elif current_length < target_length:
                            # 补零
                            padding_length = target_length - current_length
                            if np_array.ndim == 1:
                                np_array = np.pad(np_array, (0, padding_length), mode='constant', constant_values=0)
                            else:
                                padding = np.zeros((padding_length, np_array.shape[1]))
                                np_array = np.vstack((np_array, padding))

                    # 进行指定时间步的均值化操作
                    if np_array.ndim == 1:
                        # 一维数组处理
                        new_array = np.mean(np_array.reshape(-1, timesteps), axis=1)
                    else:
                        # 二维数组处理
                        new_shape = (new_length, np_array.shape[1])
                        reshaped_array = np_array[:new_length * timesteps].reshape(new_shape[0], timesteps, new_shape[1])
                        new_array = np.mean(reshaped_array, axis=1)

                    new_data.append(new_array)
                    labels.append(int(label_str))  # 记录当前样本对应的标签

                    if a % 99 == 0:
                        plt.plot(new_array.ravel())
                       # plt.show()  # 显示所有绘制的图形

                    # if new_array.size > 0:
                    #     print(f"标签 {label_str} 的文件 {pkl_file_name} 均值化后的样本:")
                    #     print(new_array.shape)
                    # else:
                    #     print(f"标签 {label_str} 的文件 {pkl_file_name} 为空，没有样本。")

        except FileNotFoundError:
            print(f"未找到压缩文件 {zip_file_path}，请检查路径是否正确。")
        except Exception as e:
            print(f"处理标签 {label_str} 时出现错误: {e}")

    new_data = np.array(new_data)
    labels = np.array(labels)
    # print("处理后的数据形状:", new_data.shape)
    # print("对应的标签形状:", labels.shape)

    # 划分训练数据和测试数据
    X_train, X_test, y_train, y_test = train_test_split(new_data, labels, test_size=0.2, random_state=42)

    print("训练数据形状:", X_train.shape)
    print("训练标签形状:", y_train.shape)
    print("测试数据形状:", X_test.shape)
    print("测试标签形状:", y_test.shape)

    return X_train, X_test, y_train, y_test


import numpy as np


def lif_encoding(data, tau=20.0, threshold=1.0, reset=0.0, dt=1.0):
    # 数据维度
    num_samples, num_timestamps, num_features = data.shape

    # 初始化脉冲输出数组
    spikes = np.zeros_like(data)

    # 初始化膜电位
    membrane_potentials = np.zeros((num_samples, num_features))

    # 对每个样本进行处理
    for i in range(num_samples):
        for t in range(num_timestamps):
            # 输入电流，从数据直接获得
            input_current = data[i, t, 0]

            # 更新膜电位
            membrane_potentials[i, :] += (dt / tau) * (-membrane_potentials[i, :] + input_current)

            # 生成脉冲
            for f in range(num_features):
                if membrane_potentials[i, f] >= threshold:
                    spikes[i, t, f] = 1  # 发放脉冲
                    membrane_potentials[i, f] = reset  # 重置膜电位

    return spikes


# 假设 standardized_X_train 是你的输入数据
# standardized_X_train = np.random.randn(960, 377, 1)  # 示例数据，你应该使用你的实际数据

# 调用函数
# spikes = lif_encoding(standardized_X_train)
# print(spikes)


import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_3d_data(X):
    """
    对三维数据集进行标准化处理。

    参数:
    X (numpy.ndarray): 形状为 (样本数, 时间步数, 特征数) 的三维数据集。

    返回:
    numpy.ndarray: 标准化后的三维数据集，形状与输入一致。
    """
    # 1. 将三维数据转换为二维数据
    original_shape = X.shape
    reshaped_X = X.reshape(-1, original_shape[-1])

    # 2. 数据标准化
    scaler = StandardScaler()
    standardized_reshaped_X = scaler.fit_transform(reshaped_X)

    # 3. 数据恢复
    # 将标准化后的二维数据恢复为原来的三维形状
    standardized_X = standardized_reshaped_X.reshape(original_shape)
    return standardized_X

#
# # # 示例调用
# zip_file_path = r'D:/researchphd/for spike code/SCN_2/texturedata.zip'
# # 可以传入不同的 target_length 和 timesteps 值
# X_train, X_test, y_train, y_test = process_and_split_data(zip_file_path, target_length=37750, timesteps=10)
# #scaler = MinMaxScaler()
# # 示例用法
# print("原始 X_train数据 形状:", X_train.shape)
# standardized_X_train = standardize_3d_data(X_train)
# X_train_spikes = lif_encoding(standardized_X_train, tau=0.01, threshold=1.0, reset=0.0, dt=1)
#
#
# # 绘制这个样本的脉冲数据
# plt.figure(figsize=(25, 3))  # 宽度较大以清晰显示所有时间点
# plt.plot(X_train_spikes[0,:], drawstyle='steps-pre')  # 使用steps-pre来更好地展示离散的脉冲数据
# plt.title(f'Pulse Data Over Time for Sample {0}')
# plt.xlabel('Time Index')
# plt.ylabel('Pulse Presence (0 or 1)')
# plt.ylim(-0.1, 1.1)  # 确保Y轴只显示0和1
# plt.grid(True)  # 添加网格线以便更清楚地看到时间点
# plt.show()

# plt.plot(standardized_X_train[4])
# plt.show()
# for i in range(5):  # 绘制前5个样本
#     x = np.arange(3775)
#     y = standardized_X_train[i].ravel()
#     plt.plot(x, y, label=f'Sample {i + 1}')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Standardized Data Samples')
# plt.legend()
# plt.show()
#
# print('shape of standardized_X_train[0] is',standardized_X_train[0].shape)
# print('standardized_X_train[0] is', standardized_X_train[0])