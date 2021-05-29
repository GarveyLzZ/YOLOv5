import sys
import os
import glob
import xml.etree.ElementTree as ET

image_ids = open('VOCdevkit/working.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in image_ids:
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse("VOCdevkit/paiwuba_xml/"+image_id+".xml").getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            if obj_name != 'paiwuba':
                continue
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
print("Conversion completed!")
