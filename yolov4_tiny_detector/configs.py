class JNConfig:
    
    def __init__(self):
        self.model_name = 'Jersey Number Detector'
        self.input_size = (416, 416)  
        
        # threshold
        self.confidence = 0.5
        self.iou = 0.3
        self.text_font = 'simhei.ttf'
        
        # classes
        self.class_names = [
            'Jersey Number'
        ]

        # anchors
        self.anchors = [
#             (10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)  # yolo default anchors
#             (6,14),  (8,20), (11,24), (15,30), (19,26), (27,63)  # YouBall dataset
#             (55, 186),  (78, 172),  (70, 206),  (90, 206), (118, 216), (228, 242)  # SVHN dataset
            (6, 10), (7, 16), (10, 13), (13, 18), (20, 27), (32, 47)  # YouBall dataset 100 classes
        ]    

            

class DigitConfig():

    def __init__(self):
        self.model_name = 'Digit Detector'
        self.input_size = (320, 320)  
        
        # threshold
        self.confidence = 0.5
        self.iou = 0.3
        self.text_font = 'simhei.ttf'
        
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
#             (10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)  # yolo default anchors
#             (6,14),  (8,20), (11,24), (15,30), (19,26), (27,63)  # YouBall dataset
            (55, 186),  (78, 172),  (70, 206),  (90, 206), (118, 216), (228, 242)  # SVHN dataset
        ]    

