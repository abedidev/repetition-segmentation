from os.path import exists

def getVideoNumber(personID):
    video = 1
    while(exists("Exercises/"+str(exercise)+"/videos/V"+str(video))):
        if exists("Exercises/"+str(exercise)+"/videos/V"+str(video)+"/"+str(personID)+".xlsx"):
            return video
        video += 1
    return -1

exercise = int(input("Enter the exercise number: "))
test_videos = ['V12', 'V2', 'V33', 'V44', 'V56']

with open("Exercises/"+str(exercise)+"/master_dataset.txt", "r") as md:
    with open("Exercises/"+str(exercise)+"/train_dataset.txt", "w") as mtrain:
        with open("Exercises/"+str(exercise)+"/test_dataset.txt", "w") as mtest:
            for line in md:
                pID = line.split(":")[0]
                if "V"+str(getVideoNumber(pID)) in test_videos:
                    mtest.write(line)
                else:
                    mtrain.write(line)
        