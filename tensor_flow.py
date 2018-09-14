from __future__ import print_function
# -*- coding:utf-8 -*-
__author__ = 'SRP'

import math

from IPython import display
from matplotlib import cm,gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

#加载数据集
dataframe = pd.read_csv('california_housing_train.csv',sep=',')

#数据随机化处理/ 房屋数量以千位单位
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
dataframe['median_house_value'] /= 1000.0


def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    '''训练一个特征的线性回归模型
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data(是否洗牌数据).
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
                  (数据应该重复的时期数。没有=无限期重复)
    Returns:
      Tuple of (features, labels) for next data batch(下一个数据批处理的元组(特性、标签))'''

    #将 pandas 数据转换成np数组
    features = {key:np.array(value) for key,value in dict(features).items()}

    #构建数据集，并配置批处理/重复
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    #如果指定，洗牌数据
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    #返回下一批数据
    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels

def train_model(learning_rate,steps,batch_size,input_feature='total_rooms'):
    '''训练一个特征的线性回归模型。
    Args:
       learning_rate: A `float`, the learning rate.
       steps: A non-zero `int`, the total number of training steps. A training step
       consists of a forward and backward pass using a single batch.
       batch_size: A non-zero `int`, the batch size.
       input_feature: A `string` specifying a column from `california_housing_dataframe`
       to use as input feature.'''
    periods = 10
    steps_per_period = steps/periods

    my_feature = input_feature
    my_feature_data = dataframe[[my_feature]]
    my_label = 'median_house_value'
    targets = dataframe[my_label]

    #创建功能列
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    #创建输入函数
    training_input_fn = lambda :my_input_fn(my_feature_data,targets,batch_size=batch_size)
    prediction_input_fn = lambda :my_input_fn(my_feature_data,targets,num_epochs=1,shuffle=False)

    #使用梯度下降法作为训练模型的优化器//创建一个线性回归对象
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
    linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,
                                                    optimizer=my_optimizer)

    #设置为绘制每个周期的模型线的状态
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.title('"Learned Line by Period')
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = dataframe.sample(n=300)
    plt.scatter(sample[my_feature],sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1,1,periods)]


    #对模型进行训练，但要在循环中进行，这样我们才能定期评估
    #损失指标。
    print('Training model(训练模型中)....')
    print('RMSE(on training data):')
    root_mean_squared_errors = []
    for period in range(periods):
        #从先前状态开始训练模型
        linear_regressor.train(input_fn=training_input_fn,
                               steps=steps_per_period)
        #Take a break and compute predictions(休息一下，计算一下预测)
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        #计算损失。
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions,targets))
        print('period %02d: %0.2f' %(period,root_mean_squared_error))

        #将这段时间的损失指标添加到我们的列表中
        root_mean_squared_errors.append(root_mean_squared_error)
        #最后，跟踪权重和偏差。
        #使用一些数学方法来确保数据和线条画得整齐。
        y_extents = np.array([0,sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' %input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents,y_extents,color=colors[period])
    print('Model training finished!!!!!')

    #输出一段时间内损失指标的图表。
    plt.subplot(1,2,2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title('Root Mean Squared Error vs. Periods')
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    # plt.show()
    plt.savefig('./img/1.png')

    #输出带有校准数据的表。
    calibration_data = pd.DataFrame()
    calibration_data['predictions'] = pd.Series(predictions)
    calibration_data['targets'] = pd.Series(targets)
    display.display(calibration_data.describe())

    print('Final RMSE (on training data): %0.2f' % root_mean_squared_error)

if __name__ == '__main__':

    # train_model(learning_rate=0.00002,
    #             steps=500,
    #             batch_size=5)

    train_model(learning_rate=0.00002,
                steps=1000,
                batch_size=5,
                input_feature='population'
                )











