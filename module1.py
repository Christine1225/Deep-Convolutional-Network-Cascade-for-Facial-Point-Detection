import Function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import random

def ReadData(Path,File):
    GtList = []
    ImageList = []
    f = open(Path + File)
    line = f.readline()
    Step = 0
    while line:
        stringList = line.split()
        FilePath = Path + stringList[0]
        BoxX = int(stringList[1])
        BoxW = int(stringList[2]) - BoxX
        BoxY = int(stringList[3])
        BoxH = int(stringList[4]) - BoxY
        PointList = [float(stringList[5]),float(stringList[6]),float(stringList[7]),float(stringList[8]),float(stringList[9]),float(stringList[10]),float(stringList[11]),float(stringList[12]),float(stringList[13]),float(stringList[14])]

        SrcImage = cv2.imread(FilePath)
        TestImage = SrcImage[int(BoxY):int(BoxY + BoxH),int(BoxX):int(BoxX + BoxW)]
        TestImage = cv2.cvtColor(TestImage,cv2.COLOR_BGR2GRAY)
        TestImage = cv2.resize(TestImage,(39,39))


        for I in range(5):
            PointList[2 * I] = (PointList[2 * I] - BoxX) / BoxW
            PointList[2 * I + 1] = (PointList[2 * I + 1] - BoxY) / BoxH 

        GtList.append(PointList)
        ImageList.append(TestImage)
        line = f.readline()
    print("Read End!",len(GtList))
    return GtList,ImageList

x = tf.placeholder(tf.float32, shape=[None, 39 , 39], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10],name='y')

x_image = tf.reshape(x, [-1, 39, 39, 1])
ConvLayer_1,ConvWeights_1 = Function.new_conv_layer(x_image,1,4,20)
print(ConvLayer_1)

ConvLayer_2,ConvWeights_2 = Function.new_conv_layer(ConvLayer_1,20,3,40)
print(ConvLayer_2)

ConvLayer_3,ConvWeights_3 = Function.new_conv_layer(ConvLayer_2,40 ,3,60)
print(ConvLayer_3)

ConvLayer_4,ConvWeights_4 = Function.new_conv_layer(ConvLayer_3,60 ,2,80,use_pooling=False)
print(ConvLayer_4)

FlatLayer,FeaturesNum = Function.flatten_layer(ConvLayer_4)
print(FlatLayer)

FcLayer1 = Function.new_fc_layer(FlatLayer,FeaturesNum,120)
print(FcLayer1)

FcLayer2 = Function.new_fc_layer(FcLayer1,120,10)
print(FcLayer2)

Cost = tf.reduce_mean(tf.pow(y * 39 - FcLayer2 * 39,2))
print(Cost)

Optimizer = tf.train.AdamOptimizer(1e-4).minimize(Cost)

GtList,ImageList = ReadData("C:\\Users\\ZhaoHang\\Documents\\Visual Studio 2017\\Projects\\PythonApplication1\\PythonApplication1\\FaceData\\","trainImageList.txt")
TestGtList,TestImageList = ReadData("C:\\Users\\ZhaoHang\\Documents\\Visual Studio 2017\\Projects\\PythonApplication1\\PythonApplication1\\FaceData\\","testImageList.txt")


session = tf.Session()
session.run(tf.global_variables_initializer())

def plot_conv_weights(weights, input_channel=0):
    # weights_conv1 or weights_conv2.

    # 运行weights以获得权重
    w = session.run(weights)

    # 获取权重最小值最大值，这将用户纠正整个图像的颜色密集度，来进行对比
    w_min = np.min(w)
    w_max = np.max(w)

    # 卷积核树木
    num_filters = w.shape[3]

    # 需要输出的卷积核
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < num_filters:
            # 获得第i个卷积核在特定输入通道上的权重
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_conv_layer(layer, image):
    # layer_conv1 or layer_conv2.

    # feed_dict只需要x，标签信息在此不需要.
    feed_dict = {x: [image]}

    # 获取该层的输出结果
    values = session.run(layer, feed_dict=feed_dict)

    # 卷积核树木
    num_filters = values.shape[3]

    # 每行需要输出的卷积核网格数
    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        # 只输出有用的子图.
        if i < num_filters:
            # 获取第i个卷积核的输出
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')

        # 移除坐标.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
              interpolation='nearest',
              cmap='binary')

    plt.show()

saver = tf.train.Saver()


#saver.restore(session,"ModelSave/Model.ckpt")
#Predict = session.run(FcLayer2,feed_dict={x:ImageList})
#print(Predict)
#for i in range(10000):
#    for p in range(5):
#        cv2.circle(ImageList[i],(int(39 * GtList[i][p * 2]),int(39 *
#        GtList[i][p * 2 + 1])),1,(255,255,255))
#        cv2.circle(ImageList[i],(int(39 * Predict[i][p * 2]),int(39
#        *Predict[i][p * 2 + 1])),1,(255,255,255))
#    cv2.imshow("test",ImageList[i])
#    cv2.waitKey(1000)
CountEq = 0
while True:
    session.run(Optimizer,feed_dict = {x:ImageList,y:GtList})
    ErrTrainSet = session.run(Cost,feed_dict={x:ImageList,y:GtList})
    ErrTestSet = session.run(Cost,feed_dict={x:TestImageList,y:TestGtList})
    print("TrainSetErr : ",ErrTrainSet," TestSetErr : ",ErrTestSet)
    CountEq += 1

    if(CountEq == 100):
        saver.save(session,str(ErrTestSet) + "/Model.ckpt")
        CountEq = 0
        
plot_conv_weights(weights = ConvWeights_1)
plot_conv_layer(layer=ConvLayer_1,image=ImageList[0])
plot_conv_layer(layer=ConvLayer_1,image=ImageList[5])

plot_conv_layer(layer=ConvLayer_2,image=ImageList[0])
plot_conv_layer(layer=ConvLayer_2,image=ImageList[5])

plot_conv_layer(layer=ConvLayer_3,image=ImageList[0])
plot_conv_layer(layer=ConvLayer_3,image=ImageList[5])

plot_conv_layer(layer=ConvLayer_4,image=ImageList[0])
plot_conv_layer(layer=ConvLayer_4,image=ImageList[5])