from textureforSNN import process_and_split_data,standardize_3d_data
#gpu or not
import torch
import time
from FBDNN import SCN
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tensorflow
#from plotmatrix import plot_confusion_matrix
#from datagenaration import datasetgenaration
from DealDataset import datasetpr, DealDataset

#from TextDataset import TextFolder
#from plot_special_data1 import plot_many_variables
import pandas as pd

import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms
# 对于实数值数据
X_train, X_test, Y_train, Y_test = process_and_split_data(
    r'texturedata.zip', target_length=37750, timesteps=10)
# X_train = convert_to_spikes(X_train.squeeze(), timesteps=10)
# X_test = convert_to_spikes(X_test.squeeze(), timesteps=10)
X_train = standardize_3d_data(X_train)
X_test = standardize_3d_data(X_test)
classes = 12
#X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Y_train 转换为 int64 类型用于 one - hot 编码
Y_train = torch.tensor(Y_train, dtype=torch.int64)-1

#X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test= torch.tensor(Y_test, dtype=torch.int64)-1


Y_zeros = torch.zeros(Y_train.shape[0] , classes)
Y_train = Y_zeros.scatter_(1, torch.tensor(Y_train.unsqueeze(1), dtype=torch.int64), 1)

Y_test_zeros = torch.zeros(Y_test.size(0), classes)
Y_test = Y_test_zeros.scatter_(1, torch.tensor(Y_test.unsqueeze(1), dtype=torch.int64), 1)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
#print(f"X_test[0]: {X_test[0,:]}")
print(f"Y_test shape: {Y_test.shape}")

print('data for training ...')

# Parameter Setting
L_max = 200# maximum hidden node number
tol = 0.003

# training tolerance
T_max = 100                    # maximun candidate nodes number

Lambdas = [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200] # scope sequence

r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]  # 1-r contraction sequence
nB = 3 # batch size
verbose = 3
use_gpu = torch.cuda.is_available()

# Model Initialization
M = SCN(L_max, T_max, tol, Lambdas, r, nB, verbose,alpha=0.25,beta = 1)

#mat = scipy.io.loadmat('Demo_Iris.mat')

T = Y_train
X = X_train
T2 = Y_test
X2 = X_test
class_names = np.array(['1', '2', '3','4','5','6','7','8','9','10'])
#class_names = np.load('selected_classes_20_3.npy')
#X3 = np.expand_dims(X,2)
#print(X3.shape)

start = time.time()
ErrorList, RateList, RateList2, timeList, suitW, suitU, Beta = M.classification(X, T, X2, T2)
end = time.time()
np.save('suitW_texture.npy', suitW)
np.save('suitU_texture.npy', suitU)
print("Time taken = ", (end - start) / 3600)
#保存模型
import pickle
pickle.dump(M,open("randommodel.dat","wb"))
np.save('error_texture.npy', ErrorList)
np.save('rate_texture.npy', RateList)
np.save('rate_te_texture.npy', RateList2)
np.save('time_texture.npy', timeList)

#导入模型
load_model = pickle.load(open("randommodel.dat","rb"))

#使用新导入的模型
(ErrorB, CMB) = load_model.getAccuracy(X2,T2)


# plot result
(ErrorA, CMA) = M.getAccuracy(X, T)

plt.figure()


#plot_confusion_matrix(CMA, classes=class_names,
                      #title='Confusion matrix, Training')

#(ErrorB, CMB) = M.getAccuracy(X2, T2)

plt.figure()

#plot_confusion_matrix(CMB, classes=class_names,
                      #title='Confusion matrix, Testing')
np.save('CMB_10_1_100%.npy', CMB)
np.save('CMA_10_1_100%.npy', CMA)
ErrorList = np.load('error10_1_100%.npy')
RateList = np.load('rate10_1_100%.npy')
confusion_matrix = np.load('CMB_10_1_100%.npy')
# Calculate the classification rate (accuracy)
accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
print(f"Classification Rate (Accuracy): {accuracy * 100:.2f}%")


plt.figure()
fig, ax1 = plt.subplots()
color1 = 'tab:red'
ax1.set_ylabel('RMSE', color = color1)
ax1.plot(range(0, ErrorList.shape[1]),
         ErrorList.reshape(-1, 1).tolist(), 'r.-')
ax1.legend(['Training RMSE'])

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('ACC', color = color2)
ax2.plot(range(0, RateList.shape[1]),
         RateList.reshape(-1, 1).tolist(), 'b.-')
ax2.plot(range(0, RateList2.shape[1]),
         RateList2.reshape(-1, 1).tolist(), 'g.-')
ax2.axis(ymin = 0, ymax = 1)
ax2.legend(['Training ACC', 'Test ACC'])

plt.show()
