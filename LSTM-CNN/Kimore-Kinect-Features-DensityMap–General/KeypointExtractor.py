from os.path import exists
import PoseDetector
import cv2

exercise = 0
while(exists("Exercises/"+str(exercise))):
    video = 1
    while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4")):
        pd = PoseDetector.poseDetector("V")
        cap = cv2.VideoCapture("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/V"+str(video)+".mp4")
        #pTime = 0
        with open("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/Keypoints.txt", "w") as k:
            while True:
                success, img = cap.read()
                if(str(img) == "None"):
                    break;

                img = pd.findPose(img)
                lmList = pd.findPosition(img)
                #print(lmList)
                if len(lmList) != 0:
                    k.write(str(lmList)+"\n")

                #cTime = time.time()
                #fps = 1/(cTime-pTime)
                #pTime = cTime
                #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) 

                #cv2.imshow("Video", img)
                #cv2.waitKey(10)
        video += 1
    exercise += 1