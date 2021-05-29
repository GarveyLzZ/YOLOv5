from keras.layers import Input
from frcnn import FRCNN
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from utils.anchors import get_anchors
from utils.utils import BBoxUtility
from nets.frcnn_training import get_new_img_size
import math
import copy
import numpy as np
import os
class mAP_FRCNN(FRCNN):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, clip_img, expand):
        self.confidence = 0.05

        image_shape = np.array(np.shape(clip_img)[0:2])
        clip_width = image_shape[1]
        clip_height = image_shape[0]
        old_image = copy.deepcopy(clip_img)
        width,height = get_new_img_size(clip_width,clip_height)


        clip_img = clip_img.resize([width,height])
        photo = np.array(clip_img,dtype = np.float64)

        # 图片预处理，归一化
        photo = preprocess_input(np.expand_dims(photo,0))
        preds = self.model_rpn.predict(photo)
        # 将预测结果进行解码
        anchors = get_anchors(self.get_img_output_length(width,height),width,height)

        rpn_results = self.bbox_util.detection_out(preds,anchors,1,confidence_threshold=0)
        R = rpn_results[0][:, 2:]
        
        R[:,0] = np.array(np.round(R[:, 0]*width/self.config.rpn_stride),dtype=np.int32)
        R[:,1] = np.array(np.round(R[:, 1]*height/self.config.rpn_stride),dtype=np.int32)
        R[:,2] = np.array(np.round(R[:, 2]*width/self.config.rpn_stride),dtype=np.int32)
        R[:,3] = np.array(np.round(R[:, 3]*height/self.config.rpn_stride),dtype=np.int32)
        
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        base_layer = preds[2]
        
        delete_line = []
        for i,r in enumerate(R):
            if r[2] < 1 or r[3] < 1:
                delete_line.append(i)
        R = np.delete(R,delete_line,axis=0)
        
        bboxes = []
        probs = []
        labels = []
        for jk in range(R.shape[0]//self.config.num_rois + 1):
            ROIs = np.expand_dims(R[self.config.num_rois*jk:self.config.num_rois*(jk+1), :], axis=0)
            
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//self.config.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.config.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            
            [P_cls, P_regr] = self.model_classifier.predict([base_layer,ROIs])

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < self.confidence or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                label = np.argmax(P_cls[0, ii, :])

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= self.config.classifier_regr_std[0]
                ty /= self.config.classifier_regr_std[1]
                tw /= self.config.classifier_regr_std[2]
                th /= self.config.classifier_regr_std[3]

                cx = x + w/2.
                cy = y + h/2.
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                x1 = cx1 - w1/2.
                y1 = cy1 - h1/2.

                x2 = cx1 + w1/2
                y2 = cy1 + h1/2

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                bboxes.append([x1,y1,x2,y2])
                probs.append(np.max(P_cls[0, ii, :]))
                labels.append(label)

        if len(bboxes)==0:
            return old_image
        
        # 筛选出其中得分高于confidence的框
        labels = np.array(labels)
        probs = np.array(probs)
        boxes = np.array(bboxes,dtype=np.float32)
        boxes[:,0] = boxes[:,0]*self.config.rpn_stride/width
        boxes[:,1] = boxes[:,1]*self.config.rpn_stride/height
        boxes[:,2] = boxes[:,2]*self.config.rpn_stride/width
        boxes[:,3] = boxes[:,3]*self.config.rpn_stride/height
        results = np.array(self.bbox_util.nms_for_out(np.array(labels),np.array(probs),np.array(boxes),self.num_classes-1,0.4))
        
        top_label_indices = results[:,0]
        top_conf = results[:,1]
        boxes = results[:,2:]
        boxes[:,0] = boxes[:,0]*clip_width  + expand[0]
        boxes[:,1] = boxes[:,1]*clip_height + expand[1]
        boxes[:,2] = boxes[:,2]*clip_width  + expand[0]
        boxes[:,3] = boxes[:,3]*clip_height + expand[1]

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)-1]
            score = str(top_conf[i])

            left, top, right, bottom = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))


        return 

frcnn = mAP_FRCNN()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

image_path = 'E:/fasterrcnn/faster-rcnn-keras-master/faster-rcnn-keras-master/VOCdevkit/paiwuba/'
image_txt  = 'E:/fasterrcnn/faster-rcnn-keras-master/faster-rcnn-keras-master/VOCdevkit/working.txt'

img_stride = 600   # 在原图上分块的步长
fill_size = 300    # w/h不足300时候就把w/h补到300
num = 0

with open(image_txt) as object:
    lines = object.readlines()


for line in lines:
    try:
        image = Image.open(image_path + line.rstrip() + '.jpg')
    except:
        print('Open Error! Try again!')
    else:
        # image.save("./input/images-optional/" + line.rstrip() + ".jpg")
        f = open("./input/detection-results/" + line.rstrip() + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        current_x = 0
        current_y = 0
        while ((current_y + img_stride) <= old_height):
            current_x = 0
            while ((current_x + img_stride) <= old_width):
                x1 = current_x
                x2 = x1 + img_stride
                y1 = current_y
                y2 = y1 + img_stride
                expand = np.array([x1, y1, x2, y2])
                clip_img = image.crop((expand[0], expand[1], expand[2], expand[3]))
                frcnn.detect_image(clip_img, expand)
                current_x += img_stride

            if  current_x < old_width:
                if (old_width - current_x) <= fill_size:
                    x1 = old_width - fill_size
                    x2 = old_width
                    y1 = current_y
                    y2 = y1 + img_stride
                else:
                    x1 = current_x
                    x2 = old_width
                    y1 = current_y
                    y2 = y1 + img_stride
                expand = np.array([x1, y1, x2, y2])
                clip_img = image.crop((expand[0], expand[1], expand[2], expand[3]))
                frcnn.detect_image(clip_img, expand)
            current_y += img_stride

        if  current_y < old_height:
            current_x = 0
            while ((current_x + img_stride) <= old_width):
                if (old_height - current_y < fill_size):
                    x1 = current_x
                    x2 = x1 + img_stride
                    y1 = old_height - fill_size
                    y2 = old_height
                else:
                    x1 = current_x
                    x2 = x1 + img_stride
                    y1 = current_y
                    y2 = old_height
                expand = np.array([x1, y1, x2, y2])
                clip_img = image.crop((expand[0], expand[1], expand[2], expand[3]))
                frcnn.detect_image(clip_img, expand)
                current_x += img_stride

            if  current_x < old_height:
                if (old_width - current_x) < fill_size:
                    x1 = old_width - fill_size
                    x2 = old_width
                else:
                    x1 = current_x
                    x2 = old_width
                if (old_height - current_y) < fill_size:
                    y1 = old_height - fill_size
                    y2 = old_height
                else:
                    y1 = current_y
                    y2 = old_height
                expand = np.array([x1, y1, x2, y2])
                clip_img = image.crop((expand[0], expand[1], expand[2], expand[3]))
                frcnn.detect_image(clip_img, expand)
        f.close()
        num += 1
        print('num = ' + str(num) + ' img = ' + line.rstrip() + '.jpg')


print("Conversion completed!")
