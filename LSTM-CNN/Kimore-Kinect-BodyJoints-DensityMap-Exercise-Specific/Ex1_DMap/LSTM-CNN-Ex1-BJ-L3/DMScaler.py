from os.path import exists

isScaled = False
if not exists("ScalingStatus.txt"):
    isScaled = False
else:
    with open("ScalingStatus.txt", 'r') as ss:
        status = ss.readline().split("\n")[0].split(" = ")[1]
        if status == "False":
            isScaled = False
        else:
            isScaled = True

action = int(input("Choose actions 1.Scale 2.RevertToOriginal: "))

if action == 2:
    if isScaled == False:
        print("Already reverted to original DMs!")
    else:
        scaling_factor = "DUMMY"
        with open("ScalingStatus.txt", 'r') as ss:
            ss.readline()
            scaling_factor = float(ss.readline().split("\n")[0].split(" = ")[1])
            exercise = 0
            while(exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps")):
                video = 1
                while(exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv")):
                    dMap = ""
                    with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", 'r') as dm:
                        dMap = dm.readline().split("\n")[0].split(",")
                        dMap = [str(float(x)/scaling_factor) for x in dMap]
                        dMap = ",".join(dMap)
                    with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "w") as dm:
                        dm.write(dMap+"\n")
                    video += 1
                exercise += 1
        with open("ScalingStatus.txt", "w") as ss:
            ss.write("Status = False\n")
            ss.write("Scaling_factor = "+str(1.0)+"\n")
else:
    prevScalingFactor = "DUMMY"
    if isScaled == False:
        prevScalingFactor = 1.0
    else:
        with open("ScalingStatus.txt", 'r') as ss:
            ss.readline()
            prevScalingFactor = float(ss.readline().split("\n")[0].split(" = ")[1])
        
    scaling_factor = float(input("Enter the scaling factor: "))
    netScalingFactor = scaling_factor * prevScalingFactor
    exercise = 0
    while(exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps")):
        video = 1
        while(exists("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv")):
            dMap = ""
            with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", 'r') as dm:
                dMap = dm.readline().split("\n")[0].split(",")
                dMap = [str(float(x)*scaling_factor) for x in dMap]
                dMap = ",".join(dMap)
            with open("Annotations/Exercise_"+str(exercise)+"/DensityMaps/V"+str(video)+"_DM.csv", "w") as dm:
                dm.write(dMap+"\n")
            video += 1
        exercise += 1
    with open("ScalingStatus.txt", "w") as ss:
        ss.write("Status = True\n")
        ss.write("Scaling_factor = "+str(netScalingFactor)+"\n")