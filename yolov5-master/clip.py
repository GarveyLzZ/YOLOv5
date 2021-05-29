import os
import xml.dom.minidom
import cv2 as cv
import math


ImgPath = 'E:/paiwukou/all/paiwuba/'
XmlPath = 'E:/paiwukou/all/paiwuba_xml/'  # xml文件地址
save_path = 'E:/paiwukou/all/clip/img'


def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:

        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)

        sizelist = collection.getElementsByTagName("size")
        for size in sizelist:
            width = size.getElementsByTagName('width')
            old_width = int(width[0].childNodes[0].data)
            height = size.getElementsByTagName('height')
            old_height = int(height[0].childNodes[0].data)


        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")

        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)

                expand_x = 0.3 / ((x2-x1) / old_width)
                newp_x1 = x1 - math.floor((x1)/expand_x)
                newp_x2 = x2 + math.floor((old_width - x2)/expand_x)
                x1 = x1 - newp_x1
                x2 = x2 - newp_x1
                print(x1, x2)

                expand_y = 0.3 / ((y2 - y1) / old_height)
                newp_y1 = y1 - math.floor((y1) / expand_y)
                newp_y2 = y2 + math.floor((old_height - y2) / expand_y)
                y1 = y1 - newp_y1
                y2 = y2 - newp_y1
                print(y1, y2)

                img = img[newp_y1:newp_y2, newp_x1:newp_x2]
                # cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                # cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), thickness=2)
                # cv.imshow('head', img)
                # cv.waitKey(0)
                paiwuba_clip.write('E:/fasterrcnn/faster-rcnn-keras-master/faster-rcnn-keras-master/VOCdevkit/VOC2007/JPEGImages/'
                                   + filename + ' ' + '%s,%s,%s,%s,0'%(x1, y1, x2, y2) + '\n')
                cv.imwrite(save_path + '/' + filename, img)  # save picture
                break
            break
paiwuba_clip = open('E:/paiwukou/all/clip/paiwuba_clip.txt', 'w')
draw_anchor(ImgPath, XmlPath, save_path)