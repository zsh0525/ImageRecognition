# ImageRecognition
  * 图像识别简单版，可识别规则的数字。（暂时不支持有干扰的图片）好处：只需引入PIL库及numpy库；实现原理：先训练模型，然后将需要的识别的图片与训练模型数据进行比较，取相似度最高的进行返回。（存在一定误差）。误差产生原因：模型算法基于像素点特证计算的。若像素点特征越小，识别效果越查。详细见代码

# image_recognition.py
  * 实现的流程（主要方法）：
    * pretreatment：将图片进行二值化处理，转换成0，1的二维数组。
    * image_split：分割图片中的数字
    * feature：提取数字特征，根据象限划分数字特征。
    * training：训练图片。并保存为pk文件(其实一种安全的是json文件)
    * distance：根据象限值计算出一个值，判断图片中数字的相似度。

# train_data
  * 基于imageTest训练后的数据。
# imageTest
   * 测试图片文件，可自行更换为风格类似的图片，即不能有干扰因素。
