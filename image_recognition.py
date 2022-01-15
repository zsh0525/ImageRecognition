# coding=utf8
from PIL import Image
import numpy
import pickle
import os
import warnings

warnings.filterwarnings('error')


class ImageRecognition():
    '''
        训练丶分析图片
    '''

    def __init__(self):
        pass

    # 区域分割
    def image_split(self, im,type=True):
        '''
        :return 各图片特征，list: 
        '''
        x_1 = []
        x_end = []
        # print im.shape[0] 获取灰度值0
        # print im.shape[1] 获取灰度值1
        # 若第一列存在1，则x添加0 说明起始位置存在1，则从0位置开始切割 竖向切割 对应X坐标
        if any(im[:, 0] == 1):
            x_1.append(0)
        for i in range(im.shape[1] - 1):  # 目的:竖向切割，在原始点中进行
            if all(im[:, i]) == 0 and any(im[:, i + 1] == 1):  # 记录分割前一个为0后一个为1的位置坐标
                x_1.append(i + 1)
            elif all(im[:, i + 1] == 0) and any(im[:, i] == 1):  # 记录分割为0前一个为1后一个为0的位置坐标
                x_end.append(i + 1)
        # all，any循环判断的高阶用法
        # 若最后一列存在1，则说明最后一列也要算。竖向切割完毕,即可得到区间 如116 分割后可得到第一区间为14-21，第二区间为36-43，第三区间为52-70

        if any(im[:, im.shape[1] - 1] == 1):
            x_end.append(im.shape[1])
        x_start = []
        area_x = 0  # 记录该截取的坐标
        # 获取截取区域的开始坐标值
        for i in x_end:
            for j in range(len(x_1)):
                if i - x_1[j] == 1:
                    x_start.append(min(x_1[area_x:j + 1]))
                    area_x = j + 1
        # 横向切割
        names = locals()  # 获取当前位置的所有局部变量
        AF = []
        for i in range(len(x_end)):
            q = im[:, range(x_start[i], x_end[i])]  # 选取竖向为1的区域
            # print q
            names['na%s' % i] = q  # 给转换后的局部变量的字典添加值。
            if any(names['na%s' % i][0, :] == 1):  # 若第一1例存在1,则y_start =0
                y_start = 0
            elif any(names['na%s' % i][names['na%s' % i].shape[0] - 1, :] == 1):  # 若最后一行例存在1,则y_end =0
                y_end = names['na%s' % i].shape[0] - 1
            for j in range(names['na%s' % i].shape[0] - 1):  # 在已选取的竖向区域进行横向切割
                if all(names['na%s' % i][j, :] == 0) and any(names['na%s' % i][j + 1, :] == 1):  # 记录前一个为0后一个为1的位置坐标
                    y_start = j + 1
                elif any(names['na%s' % i][j, :] == 1) and all(names['na%s' % i][j + 1, :] == 0):  # 记录前一个为1后一个为0的位置坐标
                    y_end = j + 1
            # 需要加去燥的判断。
            names['na%s' % i] = names['na%s' % i][range(y_start, y_end), :]  # 分割之后的区域描点
            features = self.feature(names['na%s' % i],type) # True表示返回为空。
            if len(features) != 0:
                AF.append(features)
        return AF

    # 图片二值化处理 .shape 返回矩阵的大小 .shape(x),返回长度 ,type=True:背景颜色和字体相差很大,反之背景颜色和字体相差很小，或字体的分辨小。
    def pretreatment(self, imagePath=None,type = True):
        '''
        二值化处理，转化成一个由0和1 组成的二维矩阵
        :param :图片路径: 
        :return 处理后的二维矩阵: 
        '''
        if imagePath is None:
            imagePath = self.imagePath
        ima = Image.open(imagePath)
        ima = ima.convert('L')  # 转为灰度
        # ima.show()
        im = numpy.array(ima)
        if type is True:
            threshold = (im.shape[1] + im.shape[0]) / 2  # 取灰度值
            table = []
            for i in range(256):
                if i < threshold:
                    table.append(0)
                else:
                    table.append(1)
            ima = ima.point(table, '1')  # 二值化处理 黑白
            #ima.show()
            im = numpy.array(ima)
        else:
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    if im[i, j] < 100:
                        im[i, j] = 1
                    else:
                        im[i, j] = 0
        for i in range(im.shape[0]):
            b = ""
            for j in range(im.shape[1]):
                if im[i][j] == True:
                    b = b + " 1"
                else:
                    b = b + " 0"
            # print b  # 查看特征
        return im

    # 提取图片特征
    def feature(self, A,type=True):
        '''
        根据A的像素集合，计算其分布在坐标轴的分布特征
        :param A:各区域的像素集合
        :return 特征集合: 
        '''
        try:
            midx = int(A.shape[1] / 2) + 1  # 竖向的中间位置
            midy = int(A.shape[0] / 2) + 1  # 横向的中间位置
            A1 = A[0:midy, 0:midx].mean()  # 第二象限的位置。
            A2 = A[midy:A.shape[0], 0:midx].mean()  # 第四象限的值
            A3 = A[0:midy, midx:A.shape[1]].mean()  # 第一象限的值
            A4 = A[midy:A.shape[0], midx:A.shape[1]].mean()  # 第三象限的值
            A5 = A.mean()  # 整个区域的值
            AF = [A1, A2, A3, A4, A5]
        except Warning, e:
            AF =  [9999,999,999,999,999]
            if type is True:
                AF =[]
        return AF
    # 计算两向量的距离
    def distance(self, v1, v2):
        vector1 = numpy.array(v1)
        vector2 = numpy.array(v2)
        Vector = (vector1 - vector2) ** 2
        distance = Vector.sum() ** 0.5
        return distance

    # 获取训练数据
    def get_train_data(self, dataFile):
        train_data = pickle.load(dataFile)
        return train_data

    # 计算与训练集各数据的距离集合
    def get_number(self, test_data_AF, train_data):
        result_num = ""
        for AF in test_data_AF:
            d_list = []
            for i in train_data:
                dis_dict = {}
                for key in i.keys():
                    d = self.distance(AF, i[key])
                    dis_dict[key] = d
                dis_dict = sorted(dis_dict.items(), key=lambda x: x[1], reverse=False)
            d_list.append(dis_dict[0])
            d_list = sorted(d_list, key=lambda x: x[1], reverse=False)
            distance_str = d_list[0][0]
            if "_" == distance_str:
                distance_str = "."
            result_num = result_num + str(distance_str)
        if "." in result_num:
            return float(result_num)
        if "o" in result_num:
            return int(str(result_num).replace("o", "4"))
        try:
            return int(result_num)
        except Exception as e:
            return None

    # 训练已知图片，并存为train_data.pkl,保存在该目录的下的train_data
    def training(self, filePath, threashold=True):
        train_set = {}
        train_list = []
        imageData = []
        out_data = ""
        for i, j, k in os.walk(filePath):
            imageData = k
        for i in imageData:
            im = self.pretreatment(filePath + "/" + i, threashold)
            AF = self.image_split(im, threashold)
            number = i.split(".")
            # 切割并提取特征
            for j in range(len(number[0])):
                # train_set[number[0][j]] = AF[j][0]
                train_set[number[0][j]] = AF[j]
            train_list.append(train_set)
        out_data = train_list
        # 把训练结果存为永久文件，以备下次使用
        if "_money_" in filePath:
            output = open('train_data/train_money_data.pkl', 'wb')
        elif "_game_" in filePath:
            output = open('train_data/train_game_data.pkl', 'wb')
        else:
            output = open('train_data/train_num_data.pkl', 'wb')
        pickle.dump(out_data, output)
        output.close()
        print "训练数据生成成功"

    def training12(self, filePath, threashold=True):
        train_set = {}
        train_list = []
        imageData = []
        out_data = ""
        for i, j, k in os.walk(filePath):
            imageData = k
        for i in imageData:
            im = self.pretreatment(filePath + "/" + i, threashold)
            AF = self.image_split(im, threashold)
            number = i.split(".")
            # 切割并提取特征
            for j in range(len(number[0])):
                # train_set[number[0][j]] = AF[j][0]
                train_set[number[0][j]] = AF[j]
            out_data = train_set
            if type is False:
                train_list.append(train_set)
                out_data = train_list
        # 把训练结果存为永久文件，以备下次使用

        if "_money_" in filePath:
            output = open('train_data/train_money_data.pkl', 'wb')
        elif "_game_" in filePath:
            output = open('train_data/train_game_data.pkl', 'wb')
        else:
            output = open('train_data/train_num_data.pkl', 'wb')
        pickle.dump(out_data, output)
        output.close()
        print "训练数据生成成功"

        # 计算与训练集各数据的距离集合

    def get_number12(self, test_data_AF, train_data,type =True):
        result_num = ""
        if type is True:
            for AF in test_data_AF:
                distance_dict = {}
                for key in train_data.keys():
                    # print key
                    d = self.distance(AF, train_data[key])
                    distance_dict[key] = d
                # print distance_dict
                distance_dict = sorted(distance_dict.items(), key=lambda x: x[1], reverse=False)
                distance_str = distance_dict[0][0]
                if "_" == distance_str:
                    distance_str = "."
                result_num = result_num + str(distance_str)
        else:
            for AF in test_data_AF:
                d_list = []
                for i in train_data:
                    dis_dict = {}
                    for key in i.keys():
                        d = self.distance(AF, i[key])
                        dis_dict[key] = d
                    dis_dict = sorted(dis_dict.items(), key=lambda x: x[1], reverse=False)
                    d_list.append(dis_dict[0])
                d_list = sorted(d_list, key=lambda x: x[1], reverse=False)
                distance_str = d_list[0][0]
                if "_" == distance_str:
                    distance_str = "."
                result_num = result_num + str(distance_str)
        if "." in result_num:
            return float(result_num)
        return int(result_num)
