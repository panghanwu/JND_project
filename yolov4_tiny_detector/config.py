class Config:
    
    def __init__(self):
        self.model_name = 'YOLOv4 tiny'
        self.input_size = (416, 416)  
        
        # classes
        self.class_names = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9'
        ]

        # anchors
        self.anchors = [
            # (10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)  # yolo default anchors
            (6,14),  (8,20), (11,24), (15,30), (19,26), (27,63)
        ]    

            

class DetectorConfig(Config):

    def __init__(self):
        # get general configs
        super(DetectorConfig, self).__init__()
        # threshold
        self.confidence = 0.5
        self.iou = 0.3
        self.text_font = 'simhei.ttf'
        
        
        
class TrainingConfig(Config):
    
    def __init__(self):
        # get general configs
        super(TrainingConfig, self).__init__()
        pretrain_weights = None