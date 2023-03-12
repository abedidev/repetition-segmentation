from FeatureMappings import KinectToMediaPipe as mappings
from os.path import exists
import os

exercise = 0
while(exists("Exercises/"+str(exercise))):
    with open("Exercises/"+str(exercise)+"/EssentialFeatures.csv", "r") as ef:
        with open("Exercises/"+str(exercise)+"/Dummy.csv", "w") as d:
            for line in ef:
                for mapping in mappings:
                    old_str = mapping.split(" : ")[0]
                    new_str = mapping.split(" : ")[1].replace(',', '*')
                    if old_str in line:
                        line = line.replace(old_str, new_str)
                line = line.replace('*', ',')
                d.write(line)
    with open("Exercises/"+str(exercise)+"/EssentialFeatures.csv", "w") as ef:
        with open("Exercises/"+str(exercise)+"/Dummy.csv", "r") as d:
            for line in d:
                ef.write(line)
    os.remove("Exercises/"+str(exercise)+"/Dummy.csv")
    
    
    with open("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv", "r") as nef:
        with open("Exercises/"+str(exercise)+"/Dummy.csv", "w") as d:
            for line in nef:
                for mapping in mappings:
                    old_str = mapping.split(" : ")[0]
                    new_str = mapping.split(" : ")[1].replace(',', '*')
                    if old_str in line:
                        line = line.replace(old_str, new_str)
                line = line.replace('*', ',')
                d.write(line)
    with open("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv", "w") as nef:
        with open("Exercises/"+str(exercise)+"/Dummy.csv", "r") as d:
            for line in d:
                nef.write(line)
    os.remove("Exercises/"+str(exercise)+"/Dummy.csv")
    exercise += 1