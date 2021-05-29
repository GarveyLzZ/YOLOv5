import os
import random
import numpy as np
import shutil

# 把testsample.txt放在paiwuba图片文件夹下面

# paiwuba图片路径
filepath= r"D:\BaiduNetdiskDownload\all_classification\paiwuba_xml/"
read_filepath = filepath + "testsample.txt"

# 测试样本存放文件夹路径
target = r"D:\BaiduNetdiskDownload\all_classification\test_samples_xml"


def test_variable(var):
    print(type(var))
    print(np.shape(var))
    print(var)


with open(read_filepath, "r") as f:
    data = f.readlines()
    test_variable(data)


for i in range(len(data)):
    data[i]=data[i].strip('\n')
    x = os.path.splitext(data[i])[0]+'.xml'
    source = filepath + x
    source = source.strip('\n')
    print(source)
    shutil.copy(source, target)
    os.remove(source)

