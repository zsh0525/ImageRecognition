# ImageRecognition
图像识别简单版，可识别规则的数字。（暂时不支持有干扰的图片）好处：只需引入PIL库及numpy库；实现原理：先训练模型，然后将需要的识别的图片与训练模型数据进行比较，取相似度最高的进行返回。（存在一定误差）。误差产生原因：模型算法基于像素点特证计算的。若像素点特征越小，识别效果越查。详细见代码
