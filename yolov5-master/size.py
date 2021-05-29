import xml.etree.ElementTree as ET
import os
import shutil

path_img = 'paiwuba'
path_xml = 'paiwuba_xml'
a = 0
bw_all = 0
bh_all = 0
w_all = 0
h_all = 0
i = 0
paiwuba_size = open('paiwuba_size.txt', 'w')

for roots, dirs, files in os.walk(path_xml):
    for file in files:
        i += 1
        in_file = open(os.getcwd() + '/' + path_xml + '/' + file, encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        img_size = root.find('size')
        img_width = int(img_size.find('width').text)
        w_all += img_width
        img_height = int(img_size.find('height').text)
        h_all += img_height
        paiwuba_size.write("-------------------------------------------------------------------------------------------------------------------------->%s\n"%str(i))
        paiwuba_size.write(file + ": \n")

        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            box_width = int(xmlbox.find('xmax').text) - int(xmlbox.find('xmin').text)
            bw_all += box_width
            box_height = int(xmlbox.find('ymax').text) - int(xmlbox.find('ymin').text)
            bh_all += box_height
            paiwuba_size.write('原图高(锚框高) = ' + str(img_height) + '( %s )'%str(box_height) + '\t'
                               + '原图宽(锚框宽) = ' + str(img_width) + '( %s )'%str(box_width) + '\t'
                               + '原图面积(锚框面积) = ' + str(img_height * img_width) + '( %s )'%str(box_height * box_width) + '\t\n')
            paiwuba_size.write('锚框高占比 : {:.2%}'.format(box_height / img_height) + '                    '
                               + '锚框宽占比 : {:.2%} '.format(box_width / img_width) + '                    '
                               + '锚框面积占比 : {:.2%} '.format((box_height * box_width) / (img_height * img_width)) + '\t\n')

print('原图平均高: ' + str(h_all / 8291))
print('原图平均宽: ' + str(w_all / 8291))

print('标注框平均高: ' + str(bh_all / 10912))
print('标注框平均宽: ' + str(bw_all / 10912))

print('平均高百分比: {:.2%}'.format(bh_all / h_all))
print('平均宽百分比: {:.2%}'.format(bw_all / w_all))



