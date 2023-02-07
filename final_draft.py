import cv2 #cv2 library: used to capture and process images
import numpy as np
import pyttsx3#python text to speech
import os
import threading as thread #for multithreading
import pytesseract #OCR library

engine =pyttsx3.init()
engine.setProperty('rate' , 100) #engine library used to set the text to speech library

#............................................setting up the models

current_wd = os.getcwd() #current working directory of the application 
names = ['danielle', 'danielle', 'danielle', 'Goodn', 'Goodn', 'Goodn', 'Goodn', 'hezron', 'hezron', 'hezron', 'hilda', 'hilda', 'hilda', 'jemo', 'jemo', 'jemo', 'jemo', 'junes', 'junes', 'matta', 'matta', 'matta', 'steph', 'steph', 'steph', 'zack', 'zack']
face_recognizer_model = os.path.join(current_wd, 'trained1.yml') #model used for facial recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(face_recognizer_model)
faceCascade = cv2.CascadeClassifier(os.path.join(current_wd , 'harr.xml.txt'))
font = cv2.FONT_HERSHEY_SIMPLEX


face_proto = os.path.join(current_wd , 'New folder (3)\opencv_face_detector.pbtxt.txt')#.....................
face_model = os.path.join(current_wd , 'New folder (3)\opencv_face_detector_uint8.pb')#......................

age_proto =os.path.join(current_wd , 'New folder (3)\\age_deploy.prototxt.txt')#.....................model used for age prediction...............
age_model = os.path.join(current_wd , 'New folder (3)\\age_net.caffemodel')#.....................................................................


gender_proto = os.path.join(current_wd , 'New folder (3)\gender_deploy.prototxt.txt')#..............model used for gender prediction............
gender_model = os.path.join(current_wd , 'New folder (3)\gender_net.caffemodel')#...............................................................

model_mean_val = (78.42, 87.76, 114.89)
age_list = ["(0-2)", "(4-6)" , "(8-12)", "(15-20)", "(25-32)", "(38-43)" , "(48-53)", "(60-100)"]
gender_list = ["Male" , "Female"]


class V_AID:
    def __init__(self, path = 0):
        self.path= path
        if type(path) == int: #the path isnt url
            self.cam = cv2.VideoCapture(0)#r'https://www.youtube.com/watch?v=0uQqMxXoNVs')
            print('SOURCE IS CAMERA')
            self.minW =0.1*self.cam.get(3)
            self.minH = 0.1*self.cam.get(4)

        if type(path) == str:
            pass #DO STUFF YA VIDEO CAPTURE
        self.gender_net = cv2.dnn.readNet(gender_model , gender_proto)
        self.age_net = cv2.dnn.readNet(age_model , age_proto)
    def facial_rec(self):
        detected_face = 0
        print("......RUNNING FACIAL RECOGNITION ALGORITHM.....")
        while True:
            _ , img = self.cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                    gray,  
                    scaleFactor = 1.5 ,
                    minNeighbors = 5,
                    minSize = (int(self.minW), int(self.minH)),)
            for (x,y,w,h) in faces:
                num ,confidence = recognizer.predict(gray[y:y+h ,x:x+w])
                if confidence < 100:
                    name = names[num]
                    print('DETECTED PERSON NAME {}'.format(name))
                else:
                    num = "unknown"
                    print('PERSON IS UNKOWN')
                if detected_face != num:
                    detected_face = num
                    engine.say(name)
                    engine.runAndWait()
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        self.cam.release()

    def OCR(self):
        print("........RUNNING OCR ALGORITHM..............")
        while True:
            _ , img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            last_str = ''
            if text == '':
                pass
            else:
                if last_str != text:
                    print("[[[  {}  ]]]".format(text))
                    engine.say(text)
                    engine.runAndWait()
                    last_str = text
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        self.cam.release()
    def AGE_detection(self):
        detected_age = 0
        print("............RUNNING AGE DETECTION ALGORITHM...............")
        while True:
            _ , img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                    gray,  
                    scaleFactor = 1.5 ,
                    minNeighbors = 5,
                    minSize = (int(self.minW), int(self.minH)),)
            for (x,y,w,h) in faces:
                blob = cv2.dnn.blobFromImage(img[y:y+h , x:x+w] ,1.0, (227,227),model_mean_val, swapRB = False)
                self.age_net.setInput(blob)
                age_pred = self.age_net.forward()
                age = age_list[age_pred[0].argmax()]
                if detected_age != age:
                    print('DETECTED AGE IS {}'.format(age))
                    detected_age = age
                    engine.say(age)
                    engine.runAndWait()
            k = cv2.waitKey(10) & 0xff
            if k ==27:
                break
        self.cam.release()

    def Gender_detection(self):
        print("........................GENDER DETECTION ALGORITHM RUNNING.............")
        detected_gender = 0
        while True:
            _,img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                    gray,  
                    scaleFactor = 1.5 ,
                    minNeighbors = 5,
                    minSize = (int(self.minW), int(self.minH)),)
          #  cv2.imshow('sdfdf' , gray)
            for (x,y,w,h) in faces:
                blob = cv2.dnn.blobFromImage(img[y:y+h , x:x+w] ,1.0, (227,227),model_mean_val, swapRB = False)
                self.gender_net.setInput(blob)
                gender_pred = self.gender_net.forward()
                gender = gender_list[gender_pred[0].argmax()]
                if detected_gender != gender:
                    print('DETECTED GENDER IS {}'.format(gender))
                    detected_gender = gender
                    engine.say(gender)
                    engine.runAndWait()
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        self.cam.release()

    def show(self, state = 0):
        while True:
            _, img = self.cam.read()
            img = cv2.flip(img,1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if state == 0:#for OCR detection
                
                cv2.imshow('OCR running!!' , img)
            if state == 1:# for facial recognition
                faces = faceCascade.detectMultiScale(
                                                    gray,     
                                                    scaleFactor = 1.5 ,
                                                    minNeighbors = 5,
                                                    minSize = (int(self.minW), int(self.minH)),   )
                for(x,y,w,h) in faces:
                    cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 2)
                    num ,confidence = recognizer.predict(gray[y:y+h ,x:x+w])
                    if confidence < 100:
                        confidence = "accuracy {}%".format(round(100 - confidence))
                    else:
                        confdence = "accuracy {}%".format(round(100 - confidence))
                    cv2.putText(
                                gray, 
                                str(names[num]), 
                                (x+5,y-5), 
                                font, 
                                1, 
                                (255,255,255), 
                                2
                               )
                    cv2.putText(
                                gray, 
                                str(confidence), 
                                (x+5,y+h-5), 
                                font, 
                                1, 
                                (255,255,0), 
                                1
                               )
                cv2.imshow("FACIAL_REC running!!", gray)
            if state == 2:#for age detection
                    faces = faceCascade.detectMultiScale(
                            gray,  
                            scaleFactor = 1.5 ,
                            minNeighbors = 5,
                            minSize = (int(self.minW), int(self.minH)),)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 2)
                        blob = cv2.dnn.blobFromImage(img[y:y+h , x:x+w] ,1.0, (227,227),model_mean_val, swapRB = False)
                        self.age_net.setInput(blob)
                        age_pred = self.age_net.forward()
                        age = age_list[age_pred[0].argmax()]
                        cv2.putText(
                                gray, 
                                "AGE: "+str(age), 
                                (x+5,y+h-5), 
                                font, 
                                1, 
                                (255,0,0), 
                                1
                               )
                    cv2.imshow("AGE DETECTION running!!" , gray)
                
            if state == 3:#for gender detection
                    faces = faceCascade.detectMultiScale(
                            gray,  
                            scaleFactor = 1.5 ,
                            minNeighbors = 5,
                            minSize = (int(self.minW), int(self.minH)),)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 2)
                        blob = cv2.dnn.blobFromImage(img[y:y+h , x:x+w] ,1.0, (227,227),model_mean_val, swapRB = False)
                        self.gender_net.setInput(blob)
                        gender_pred = self.gender_net.forward()
                        gender = gender_list[gender_pred[0].argmax()]
                        cv2.putText(
                                gray, 
                                "GENDER: "+str(gender), 
                                (x+5,y+h-5), 
                                font, 
                                1, 
                                (255,0,0), 
                                1
                               )
                    cv2.imshow("GENDER DETECTION running!!" , gray)
                
                
                        
                
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break
        self.cam.release()

X = V_AID()

graphic = '''
                                       
                          #######          #########
                        ###     ##           ## ##
                        ###                  ## ##
                        ###   #####          ## ##
                        ###    ###           ## ##
                          ####### roup     #########

                PROJECT WAS DESIGNED BY GROUP 2. THE AIM WAS TO
                DEVELOP A VISUAL AID FOR THE BLIND.

                DEVELOPERS

                        NAME         ADMISION NUM       ALIAS
                        ____         ____________       _____

                    1. GOODING PAUL  BSMD150J2018       61M8
                    2. DORIS DANIELLE BSMD/136J/2018
                    3. HILDA JESANG'
                    4. STEPHIE ODHIAMBO
                    5. ZAKAYO EDWIN OSHOME BSMD/137J/2018
                    6. JAMES MUGAMBI
                    7. SABASTIAN ONDIEKI
                    8. HEZRON OLILA
                    9. FELIX KANJA
                    10.JUNES CHEBET
                    11. MARK MATAKILI
                    
          '''


print(graphic)
thread1 = thread.Thread(target = X.OCR)
thread2 = thread.Thread(target = X.show , args = (2,))
thread2.start()
thread1.start()
#X.OCR()
#X.AGE_detection()
#X.Gender_detection()
#X.show(state = 3)
