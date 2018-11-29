# -*- coding: utf-8 -*-
"""
简介：
建立一个预测在2D流场中预测游动细丝距离探测点的位置、游动速度和细丝刚度的神经网络模型

Created on Tue Nov 27 10:25:03 2018

@author: gear
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout
import matplotlib.pyplot as plt

import keras.backend as K

def Distance(root, r):
    '''
    计算不同刚度的细丝在不同时刻距离探测器的水平位置
    Parameters:
        root -- 文件根目录
        r -- 细丝刚度
    Returns:
        S -- 细丝距离探测器的水平位置
    '''
    path = root + 'data/r=' + str(r) + '/lag/'
    filenames=os.listdir(path)  #返回指定目录下的所有文件和目录名
    numbs = len(filenames)
    
    S = []      # 细丝头部距离探测器圆心的位置
    for i in range(numbs):
        with open(path + filenames[i]) as file:
            x = file.readline().strip().split()
            S.append(9-float(x[0]))         # 这里9指探测器圆心水平位置
    
    return S

def Input_X(root, path, y, v, r):
    '''
    读取训练数据并做预处理
    Parameters:
        path -- 训练数据所在文件路径
        y -- 探测器相对鱼的Y距离
        v -- 游动细丝的巡航速度    
        r -- 游动细丝的刚度
    Returns:
        train_X -- Array, 每行数据为16个探测点的（Ux, Uy, W), shape=(m, 48)
        train_Y -- Array, 训练标签， 每行为(S, V, R), shape=(m, 3)
        test_X -- 同train_X
        test_Y -- 同trian_Y
    '''
    filenames=os.listdir(path)  #返回指定目录下的所有文件和目录名
        
    X = []
    numbs = len(filenames)
    for i in range(0, numbs):
        path1 = path + filenames[i] 
        df = pd.read_table(path1, header=None, skiprows=[0,1,2,3,4,5,6], sep='\s+')
        df.columns = ['X', 'Y', 'Ux', 'Uy', 'W']
        data = df.drop(['X','Y'], axis=1)
        temp = data.values
        temp = temp.reshape(1,48)
        X.append(temp)
    X = np.array(X).reshape(200,48)
    
    S = Distance(root, r)                # 探测器相对鱼的X距离
    X = np.column_stack((X, S))
    
    Y = np.ones((200, 1)) * y      # 探测器相对鱼的Y距离
    X = np.column_stack((X, Y))
    
    V = np.ones((200, 1)) * v      # 游动细丝的巡航速度
    X = np.column_stack((X, V))
                    
    return X
       
def Input_data(root):
    '''
    载入所有训练数据
    Parameters:
        root -- 训练数据根目录
    Returns:
        train -- 训练数据集
        test -- 测试数据集
    '''
    R = [0.5, 1.0]      # 不同细丝的刚度
    V = [1.27, 1.87]     # 刚度对应的巡航速度
    Y_dist = [0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    nums1 = len(R)
    
    dir1 = 'H:/job_2/vortex/r=0.5/vortex_r=0.5_y=0.0/'
    dir2 = 'H:/job_2/vortex/r=1.0/vortex_r=1_y=0.0/'
    X1 = Input_X(root, dir1, y=Y_dist[0], v=V[0], r=R[0])
    X2 = Input_X(root, dir2, y=Y_dist[0], v=V[0], r=R[0])
    X = np.row_stack((X1, X2))
    
    for i in range(nums1):
        path = root + 'vortex/r=' + str(R[i]) + '/'
        filenames = os.listdir(path)
        nums = len(filenames)
        for j in range(1,nums):
            dir = path + filenames[j] + '/'
            temp = Input_X(root, dir, y = Y_dist[j], v=V[i], r=R[i])
            X = np.row_stack((X, temp))
        
    # 打乱数据集
    index = [i for i in range(len(X))]  
    np.random.shuffle(index) 
    X = X[index] 
    
    train = X[0:3100]
    test = X[3100:]
    
    return train, test

def ShowResults(test_X, test_Y, predict_Y):
    '''
    显示测试数据集对应的预测结果
    Parameters:
        test_X -- 测试数据集
        test_Y -- 测试标签
        predict_Y -- 测试集对应的预测结果
    Returns:
        None
    '''
    sns.set(color_codes=True)
    plt.plot(test_Y[:,0])
    plt.plot(predict_Y[0])
    plt.legend(['Sx_true', 'Sx_pred'])
    plt.show()
    
    plt.plot(test_Y[:,1])
    plt.plot(predict_Y[1])
    plt.legend(['Sy_true', 'Sy_pred'])
    plt.show()
    
    plt.plot(test_Y[:,2])
    plt.plot(predict_Y[2])
    plt.legend(['Vx_true', 'Vx_pred'])
    plt.ylim(0, 2.5)
    plt.show()

def FishModel(input_shape):
    '''
    实现一个预测游动细丝位置、游动速度和刚度的模型
    Parameters:
        input_shape -- 输入的数据维度，这里为input_shape=(3,1)
    Returns:
        model -- 创建的FishModel模型
    '''
    
    # 输入的训练数据
    X_input = Input(input_shape)
    
    # 第01层网络
    X = Dense(units=48, activation='relu')(X_input)
    X = Dropout(rate=1)(X)
    
    # 第02层网络
    X = Dense(units=48, activation='relu')(X)
    X = Dropout(rate=1)(X) 
    
    # 第03层网络
    X = Dense(units=48, activation='relu')(X)
    X = Dropout(rate=1)(X)
    
    # 第04层网络
    X = Dense(units=96, activation='relu')(X)
    X = Dropout(rate=1)(X)
    
    # 第05层网络
    X = Dense(units=96, activation='relu')(X)
    X = Dropout(rate=1)(X)
    
    # 第06层网络
    X = Dense(units=48, activation='relu')(X)
    X = Dropout(rate=1)(X)
    
    # 第07层网络
    X = Dense(units=48, activation='relu')(X)
    X = Dropout(rate=1)(X)
    
    # 第08层网络
    X = Dense(units=48, activation='relu')(X)
    
    # 输出层
    Sx = Dense(units=1, name='Sx')(X)  #x方向距离
    Sy = Dense(units=1, name='Sy')(X)  #y方向距离
    Vx = Dense(units=1, name='speed')(X)     #x方向速度
    

    
    model = Model(inputs=X_input, outputs= [Sx, Sy, Vx])
    
    return model

def TrainMoel(model, train_X, train_Y):
    '''
    对细丝推进模型进行训练
    Parameters:
        model -- 建立的FishModel
        train_X -- 训练数据
        train_Y -- 训练标签
    Returns:
        None
    '''
    # step1 编译模型（定义训练方式）
    model.compile(loss=['mse', 'mse', 'mse'], optimizer='adam', metrics=['accuracy'])
    
    # step2 训练模型
    model.fit(train_X, [train_Y[:,0], train_Y[:,1], train_Y[:,2]], validation_split=0.2, 
              epochs=200, batch_size=20, verbose=2)
    
    # step3 评估模型
    score = model.evaluate(train_X, [train_Y[:,0], train_Y[:,1], train_Y[:,2]])
    print('Test loss: ', score[0])

    
    return model

def Prediction(model, path):
    '''
    对任意位置探测器探测到的数据进行预测
    Parameters:
        model -- 训练的模型
        test_X -- 任意位置探测到的数据
    Returns:
        pred_Y -- 预测的值
    '''
    
    df = pd.read_table(path, header=None, skiprows=[0,1,2,3,4,5,6], sep='\s+')
    df.columns = ['X', 'Y', 'Ux', 'Uy', 'W']
    data = df.drop(['X','Y'], axis=1)
    temp = data.values
    temp = temp.reshape(1,48)
    test_X = temp
    
    pred_Y = model.predict(test_X)
    print(pred_Y)
    
    return pred_Y

def Shuffle(data):
    '''
    随机打乱数据集
    Parameters:
        data -- 待打乱数据集
    Returns:
        train -- 打乱后的训练数据集
        test -- 打乱后的测试数据集
    '''
    data = data.values
    index = [i for i in range(len(data))]  
    np.random.shuffle(index) 
    X = data[index] 
    
    train = X[0:8900]
    test = X[8900:]
    
    return train, test
    

if __name__ == '__main__':
    
    fish_data = 'H:/job_2/py_code/fish_data.csv'
    path = 'H:/job_2/test.dat'
   
    fish_data = pd.read_csv(fish_data, index_col=0)      # 探测器在细丝上方的数据
    train, test = Shuffle(fish_data)
    
    train_X = train[:,0:48]
    train_Y = train[:,48:51]
    test_X = test[:,0:48]
    test_Y = test[:,48:51]
    
    # 进行训练    
    input_shape = train_X.shape[1:] 
    fish_model = FishModel(input_shape)
    model = TrainMoel(fish_model, train_X, train_Y)
    
    # 进行预测
    predict_Y = model.predict(test_X)
    print(predict_Y)
    
    # 显示预测结果
    ShowResults(test_X, test_Y, predict_Y)
    
    # 单个位置预测
    Prediction(model, path)
    
    

            

