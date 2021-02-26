# from PIL import
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn as nn
import torch
import os
import colorsys

# local module
from .nets.yolo4_tiny import YoloBody
from .utils.utils import (
    DecodeBox, 
    non_max_suppression
)
from .config import DetectorConfig


# load configuration
config = DetectorConfig()



class Detector:
    
    def __init__(self):
        self.class_names = config.class_names
        self.input_size = config.input_size
        self.anchors = np.array(config.anchors).reshape([-1, 3, 2])
        self.confidence = config.confidence
        self.iou = config.iou
        self.text_font = config.text_font
        
        
    # set/reset model
    def init(self, weight_path, device='cpu'):
        print('Initializing model...')
        
        
        # set decive
        assert device in ['cpu', 'cuda']
        if device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.cuda = True
                print('Set device to "cuda" successfully!')
            else:
                device = torch.device('cpu')
                print('Cannot reach available "cuda". The device will set to "cpu".')
                self.cuda = False
        elif device == 'cpu':
            device = torch.device('cpu')
            self.cuda = False
            print('Set device to "cpu" successfully!')
        
        
        # load model
        print('Loading weights into state dict...')
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()
        state_dict = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(state_dict)

        # I dont know what is this.
        # Seem like parallel computation setting...
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        
        # bbox decoder
        print('Initializing YOLO bounding-box decoder...')
        self.yolo_decodes = []
        self.anchors_mask = [[3,4,5],[1,2,3]]
        for i in range(2):
            self.yolo_decodes.append(DecodeBox(
                np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], 
                len(self.class_names),  
                (self.input_size[1], self.input_size[0]))
        )
        
        # set colors for each class     
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))        
            
        print('Finished!')
        
        
    def detect_image(self, image, grayscale=False): 
        image_shape = np.array(np.shape(image)[0:2])
        # make a copy
        self.image = image.copy()

        if grayscale:
            image = image.convert('L')
        # set channels to 3
        image = image.convert('RGB')
        # resize to fit input
        image = image.resize((self.input_size[1],self.input_size[0]), Image.BICUBIC)
        # normalize
        image = np.array(image, dtype = np.float32) / 255.0
        # transpose to fit input
        image = np.transpose(image, (2, 0, 1))
        # add batch dim
        image = [image]
        
        # turn off autograd
        with torch.no_grad():
            self.result = []  # create result container
            image = torch.from_numpy(np.asarray(image))
            if self.cuda:
                image = image.cuda()
                
            # detect via model
            outputs = self.net(image)
            
            # decode bbox
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))
                
            # NMS
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(
                output, 
                len(self.class_names),
                conf_thres=self.confidence,
                nms_thres=self.iou
            )
            
            # if no object is detected
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return self.result
            
            # filter bbox under threshold
            top_index = batch_detections[:, 4]*batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index,-1], np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin = np.expand_dims(top_bboxes[:,0],-1)
            top_ymin = np.expand_dims(top_bboxes[:,1],-1)
            top_xmax = np.expand_dims(top_bboxes[:,2],-1)
            top_ymax = np.expand_dims(top_bboxes[:,3],-1)
            
            # align bbox back to origin size
            top_xmin = top_xmin/self.input_size[1] * image_shape[1]
            top_ymin = top_ymin/self.input_size[0] * image_shape[0]
            top_xmax = top_xmax/self.input_size[1] * image_shape[1]
            top_ymax = top_ymax/self.input_size[0] * image_shape[0]
            boxes = np.concatenate([top_xmin,top_ymin,top_xmax,top_ymax], axis=-1)
            
            # gather bbox to list (c, s, (top, left, bottom, right))
            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]
                score = top_conf[i]
                self.result.append([c,predicted_class,score,list(boxes[i])])
            return self.result
        
        
    def mark_image(self, image, line_thick=3, show_text=True):
        if show_text:
            text_size = np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32')
            font = ImageFont.truetype(font='simhei.ttf', size=text_size)
        
        draw = ImageDraw.Draw(image)
        # sort bboxes by score
        for bbox in sorted(self.result, key=lambda s: s[2]):
            # bbox
            draw.rectangle(bbox[3], fill=None, outline=self.colors[bbox[0]], width=line_thick)
            # tag
            if show_text:
                tag = '{} {:.2f}'.format(bbox[1], bbox[2])
                text_po = [bbox[3][0], bbox[3][1]-text_size, bbox[3][2], bbox[3][1]]
                draw.rectangle(text_po, fill=self.colors[bbox[0]], outline=self.colors[bbox[0]], width=3)
                draw.text(text_po, tag, fill=(0,0,0), font=font)
        del draw
        return image