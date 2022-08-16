import numpy as np
import tensorflow as tf
import cv2
import time
import glob, os
from notify_run import Notify
notify = Notify()
import serial #for Serial communication
import time   #for delay functions

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
#Login to Google Drive and create drive object
g_login = GoogleAuth()
g_login.LocalWebserverAuth()
drive = GoogleDrive(g_login)
# Importing os and glob to find all PDFs inside subfolder


delete_id=None 
print('Initializing Arduino')
arduino = serial.Serial(port='/dev/ttyACM0',baudrate = 9600)   #Create Serial port object called arduinoSerialData
print(arduino.readline())
print('Initialized Arduino')

#initializing Timing and mode

Mode=2
intruder=-2
    
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (0, 185) 
# fontScale 
fontScale = 1
# Red color in BGR 
color = (0, 0, 255) 
# Line thickness of 2 px 
thickness = 2
# Using cv2.putText() method 


def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()
        
if __name__=='__main__':
    model_path = 'frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.4
    cap = cv2.VideoCapture(0)

    ## Saving recorded Video
    ## 

    while True:
        r, img = cap.read()
        if r==True:
            boxes, scores, classes, num = odapi.processFrame(img)
            counter=0
            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    counter+=1
                    if intruder==0:
                        cv2.imwrite('screenshot.jpeg',img)
                        #NOTIFIER
                        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
                        for file1 in file_list:
                            print('title: %s, id: %s' % (file1['title'], file1['id']))
                            if file1['title']=='screenshot.jpeg':
                                delete_id= file1['id']
                        if delete_id:
                            file_del = drive.CreateFile({'id':delete_id})
                            file_del.Delete()

                        file_drive = drive.CreateFile({'title': 'screenshot.jpeg' }) 
                        file_drive.SetContentFile("screenshot.jpeg") 
                        file_drive.Upload()
                        file_drive.InsertPermission({'type': 'anyone','value': 'anyone','role': 'reader'})
                        print("The file: " + 'screenshot.jpeg' + " has been uploaded")
                        print("All files have been uploaded")
                        file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
                        for file1 in file_list:
                            print('title: %s, id: %s' % (file1['title'], file1['id']))
                            if file1['title']=='screenshot.jpeg':
                                fetch_id= file1['id']
                        notify.send("Caution Intrusion Detected!!",'https://drive.google.com/open?id='+fetch_id)
                        
                        ##NOTIFICATION ENDS
                        
                    intruder-=1
            print("Humans Detected: "+str(counter))
            if(counter>0):
                if Mode==2:
                    arduino.write(str.encode('2'))
                    time.sleep(5)
                else:
                    arduino.write(str.encode('1'))
                    time.sleep(5)
            else:
                arduino.write(str.encode('0'))
            cv2.imshow("Video Footage", img)
            key = cv2.waitKey(1)
            if key &0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
