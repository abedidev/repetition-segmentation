#Currently under development....

import cv2
import time

class KeypointsVisualiser(object):
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

def main():
    exercise = int(input("Enter the exercise number: "))
    video = int(input("Enter the video number: "))
    path = "Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4"
    cap = cv2.VideoCapture(path) #Replace 0 with the video filename for reading from video files
    #pTime = 0
    detector = KeypointsVisualiser()
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


            #cTime = time.time()
            #fps = 1/(cTime-pTime)
            #pTime = cTime
            #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) 

            cv2.imshow("Image", img)
            cv2.waitKey(10)
            
if __name__ == "__main__":
    main()