import cv2 
import mediapipe as mp
import time
import math

class poseDetector:
    keyMap = {  0: 'nose',
                1: 'left_eye_inner',
                2: 'left_eye',
                3: 'left_eye_outer',
                4: 'right_eye_inner',
                5: 'right_eye',
                6: 'right_eye_outer',
                7: 'left_ear',
                8: 'right_ear',
                9: 'mouth_left',
                10: 'mouth_right',
                11: 'left_shoulder',
                12: 'right_shoulder',
                13: 'left_elbow',
                14: 'right_elbow',
                15: 'left_wrist',
                16: 'right_wrist',
                17: 'left_pinky',
                18: 'right_pinky',
                19: 'left _index',
                20: 'right_index',
                21: 'left_thumb',
                22: 'right_thumb',
                23: 'left _hip',
                24: 'right_hip',
                25: 'left_knee',
                26: 'right_knee',
                27: 'left_ankle',
                28: 'right_ankle',
                29: 'left_ heel',
                30: 'right_heel',
                31: 'left_foot_index',
                32: 'right_foot_index' }
    
    def __init__(self, detectorType, detectionDimension = 2, modelComplex = 2, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplex
        self.detectionDimension = detectionDimension
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        if detectorType == "V":
            self.pose = self.mpPose.Pose(self.mode, self.upBody, self.modelComplex, self.smooth, self.detectionCon, self.trackCon)
        else:
            self.pose = self.mpPose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks) #Put this in a list for future analysis
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    
    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = (lm.x * w), (lm.y * h)
                cz = (lm.z * w)
                visibility = lm.visibility
                lmList.append([id, cx, (-1)*cy, cz, visibility])
                if draw:
                    cv2.circle(img, (int(cx), int(cy)), 5, (255, 0 , 0), cv2.FILLED)
        return lmList
    
    
    def findAngles(self, lmList):
        #Returns a list of the form: [[String::Name, Float::Phi, Float::Theta, Float::r]]
        angles = []
        visThreshold = 0.5
        aList = [[0, 4], [0, 1], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8], 
                 [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [17, 19], 
                 [15, 21], [11, 23], [12, 14], [14, 16], [16, 22], [16, 18], [18, 20], 
                 [16, 20], [12, 24], [23, 24], [23, 25], [24, 26], [26, 28], [28, 32], 
                 [28, 30], [25, 27], [27, 29], [27, 31], [29, 31], [30, 32]]
        for pr in aList:
            if lmList[pr[0]][4] < visThreshold or lmList[pr[1]][4] < visThreshold:
                continue
            
            point1 = []
            point2 = []
            
            point1.append(lmList[pr[0]][1])
            point1.append(lmList[pr[0]][2])
            point1.append(lmList[pr[0]][3])
            
            point2.append(lmList[pr[1]][1])
            point2.append(lmList[pr[1]][2])
            point2.append(lmList[pr[1]][3])
            
            angle = self.calculateAngles(point1, point2)
            angle[0] = self.keyMap[pr[0]] + "->" + self.keyMap[pr[1]]
            angles.append(angle)
        return angles
    
    def findAnglesBV(self, lmList):
        angles = []
        visThreshold = 0.5
        jList = [[16, 18, 20], [18, 20, 16], [20, 16, 18], [20, 16, 22], [22, 16, 14], [16, 14, 12], 
                 [14, 12, 24], [11, 12, 24], [12, 11, 23], [13, 11, 23], [11, 13, 15], [13, 15, 21], 
                 [21, 15, 19], [19, 15, 17], [15, 17, 19], [17, 19, 15], [12, 24, 23], [11, 23, 24], 
                 [26, 24, 23], [25, 23, 24], [28, 26, 24], [27, 25, 23], [32, 28, 26], [31, 27, 25], 
                 [32, 28, 30], [28, 30, 32], [30, 32, 28], [31, 27, 29], [27, 29, 31], [29, 31, 27]]
        for pr in jList:
            if lmList[pr[0]][4] < visThreshold or lmList[pr[1]][4] < visThreshold or lmList[pr[2]][4] < visThreshold:
                continue
            
            point1 = []
            point2 = []
            point3 = []
            
            point1.append(lmList[pr[0]][1])
            point1.append(lmList[pr[0]][2])
            point1.append(lmList[pr[0]][3])
            
            point2.append(lmList[pr[1]][1])
            point2.append(lmList[pr[1]][2])
            point2.append(lmList[pr[1]][3])
            
            point3.append(lmList[pr[2]][1])
            point3.append(lmList[pr[2]][2])
            point3.append(lmList[pr[2]][3])
            
            angle = self.calculateAnglesBV(point1, point2, point3)
            angle[0] = self.keyMap[pr[0]] + "<-" + self.keyMap[pr[1]] + "->" + self.keyMap[pr[2]]
            angles.append(angle)
        return angles
    
    def calculateAngles(self, p1, p2):
        #p1 and p2 are lists of the form: [x, y, z]
        #Returns a list of the form [String::Name, Float::Phi, Float::Theta, Float::r]
        angle = [""]
        x1 = p1[0]
        y1 = p1[1]
        z1 = p1[2]
        
        x2 = p2[0]
        y2 = p2[1]
        z2 = p2[2]
        
        X = x1 - x2
        Y = y1 - y2
        Z = z1 - z2
        
        #Calculating r:
        r = math.sqrt((X*X)+(Y*Y)+(Z*Z))
        
        #Calculating Phi:
        Phi = -1.0
        if X == 0:
            if Y == 0:
                Phi = -1.0 #Indicates that Phi is not defined in this case
            elif Y > 0:
                Phi = 90.0
            else:
                Phi = 270.0
        else:
            ratio = abs(Y/X)
            if X > 0 and Y >= 0:
                Phi = self.toDegree(math.atan(ratio))
            elif X > 0 and Y < 0:
                Phi = 360.0 - self.toDegree(math.atan(ratio))
            elif X < 0 and Y >= 0:
                Phi = 180.0 - self.toDegree(math.atan(ratio))
            else:
                Phi = 180.0 + self.toDegree(math.atan(ratio))
        
        #Calculating Theta:
        Theta = -1.0
        if r != 0:
            try:
                Theta = self.toDegree(math.acos(Z/r))
            except Exception:
                Theta = -1.0
        else:
            Phi = -1.0 #Indicates that Phi is not defined in this case
            Theta = -1.0 #Indicates that Theta is not defined in this case
        
        angle.append(Phi)
        angle.append(Theta)
        angle.append(r)
        
        return angle
    
    def calculateAnglesBV(self, point1, point2, point3):
        angle = [""]
        
        V1 = [point1[0] - point2[0], point1[1] - point2[1]]
        V2 = [point3[0] - point2[0], point3[1] - point2[1]]
        
        dotProd = V1[0]*V2[0] + V1[1]*V2[1]
        modV1 = math.sqrt(V1[0]*V1[0] + V1[1]*V1[1])
        modV2 = math.sqrt(V2[0]*V2[0] + V2[1]*V2[1])
        
        if modV1 == 0 or modV2 == 0:
            angle.append(-1.0)
        else:
            try:
                angle.append(self.toDegree(math.acos(dotProd/(modV1*modV2))))
            except Exception:
                angle.append(-1.0)
        
        return angle
    
    def printAngles(self, angles):
        #angles is a list of the form: [[String::Name, Float::Phi, Float::Theta, Float::r]]
        for i in angles:
            if i[1] == -1:
                i[1] = "Not Defined"
            else:
                i[1] = round(i[1], 2)
                
            if i[2] == -1:
                i[2] = "Not Defined"
            else:
                i[2] = round(i[2], 2)
                
            if(self.detectionDimension == 3):
                print("Name: "+i[0]+"  Phi: "+str(i[1])+"  Theta: "+str(i[2])+"  r: "+str(round(i[3], 2)))
            else:
                print("Name: "+i[0]+"  Phi: "+str(i[1])+"  r: "+str(round(i[3], 2)))
                
    def printAnglesBV(self, angles):
        #angles is a list of the form: [[String::Name, Float::Phi]]
        for i in angles:
            if i[1] == -1:
                i[1] = "Not Defined"
            else:
                i[1] = round(i[1], 2)
                
            print("Name: "+i[0]+"  Phi: "+str(i[1]))
                
    def toDegree(self, ang):
        return ang*(180/math.pi)
        
        
    
    

def main(): #Put testing script
    exercise = int(input("Enter the exercise number: "))
    video = int(input("Enter the video number: "))
    path = "Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4"
    cap = cv2.VideoCapture(path) #Replace 0 with the video filename for reading from video files
    #pTime = 0
    detector = poseDetector("V", 2) #Currently set for 2D pose detection
    start_time = time.time()
    prev_time = start_time
    while True:
        start_time = time.time()
        success, img = cap.read()
        if(str(img) == "None"):
            break;
        
        if(start_time - prev_time > 5 or True): #Dummy frame extracting if statement
            prev_time = start_time
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            #print(lmList)
            if len(lmList) != 0:
                angles = detector.findAnglesBV(lmList)
                detector.printAnglesBV(angles)
                print("\n\n")
                
            
            #cTime = time.time()
            #fps = 1/(cTime-pTime)
            #pTime = cTime

            #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) 
            
            cv2.imshow("Image", img)
            cv2.waitKey(10)
    
if __name__ == "__main__":
    main()