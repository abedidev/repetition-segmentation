from os.path import exists
import os
import collections

def hasSameElements(lst1, lst2):
    return collections.Counter(lst1) == collections.Counter(lst2)

def isSame(descriptor1, descriptor2):
    descriptor1 = descriptor1.lower().split(", ")
    descriptor2 = descriptor2.lower().split(", ")
    
    medianCheck1 = False
    medianCheck2 = False
    new_descriptor11 = ""
    new_descriptor12 = ""
    new_descriptor21 = ""
    new_descriptor22 = ""
    
    i=0
    while(i<len(descriptor1)):
        if (type(descriptor1[i]) != int) and (descriptor1[i].lower() == 'm'):
            medianCheck1 = True
            new_descriptor11 += str(int(descriptor1[i+1]) + int(descriptor1[i+2]))
            new_descriptor12 += str(abs(int(descriptor1[i+1]) - int(descriptor1[i+2])))
            if ((i+2) != (len(descriptor1) - 1)):
                new_descriptor11 += ", "
                new_descriptor12 += ", "
            i = i + 3
        else:
            new_descriptor11 += descriptor1[i]
            new_descriptor12 += descriptor1[i]
            if i != len(descriptor1) - 1:
                new_descriptor11 += ", "
                new_descriptor12 += ", "
            i += 1
          
    i=0
    while(i<len(descriptor2)):
        if (type(descriptor2[i]) != int) and (descriptor2[i].lower() == 'm'):
            medianCheck2 = True
            new_descriptor21 += str(int(descriptor2[i+1]) + int(descriptor2[i+2]))
            new_descriptor22 += str(abs(int(descriptor2[i+1]) - int(descriptor2[i+2])))
            if ((i+2) != (len(descriptor2) - 1)):
                new_descriptor21 += ", "
                new_descriptor22 += ", "
            i = i + 3
        else:
            new_descriptor21 += descriptor2[i]
            new_descriptor22 += descriptor2[i]
            if i != len(descriptor2) - 1:
                new_descriptor21 += ", "
                new_descriptor22 += ", "
            i += 1
    
    if medianCheck1 != medianCheck2:
        return False
    if (medianCheck1 == True) and (medianCheck2 == True):
        if (isSame(new_descriptor11, new_descriptor21)) and (isSame(new_descriptor12, new_descriptor22)):
            return True
        else:
            return False
                
    if (descriptor1[0] != descriptor2[0]) or (descriptor1[1] != descriptor2[1]) or (len(descriptor1) != len(descriptor2)):
        return False
    
    parameters1 = [int(x) if x.isdigit() else x for x in descriptor1[2:]]
    parameters2 = [int(x) if x.isdigit() else x for x in descriptor2[2:]]
    featureType = descriptor1[0]+descriptor1[1]
    
    if featureType == '2d':
        if len(parameters1) == 2:
            return hasSameElements(parameters1, parameters2)
        elif len(parameters1) == 4:
            p11 = parameters1[0:2]
            p12 = parameters1[2:4]
            p21 = parameters2[0:2]
            p22 = parameters2[2:4]
            if (hasSameElements(p11, p21) and hasSameElements(p12, p22)) or (hasSameElements(p11, p22) and hasSameElements(p12, p21)):
                return True
            else:
                return False
            
    elif featureType == '2k':
        if descriptor1 == descriptor2:
            return True
        else:
            return False
        
    elif featureType == '2a':
        if type(parameters1[2]) == int and type(parameters2[2]) == int:
            p1 = parameters1[0:1]+parameters1[2:3]
            p2 = parameters2[0:1]+parameters2[2:3]
            if (parameters1[1] == parameters2[1]) and (hasSameElements(p1, p2)):
                return True
            else:
                return False
        elif type(parameters1[2]) != int and type(parameters2[2]) != int:
            if parameters1[2] != parameters2[2]:
                return False
            else:
                return hasSameElements(parameters1[0:2], parameters2[0:2])
        else:
            return False
            
    
    elif featureType == '2v':
        if len(parameters1) == 1:
            if descriptor1 == descriptor2:
                return True
            else:
                return False
        
        elif len(parameters1) == 2:
            return hasSameElements(parameters1, parameters2)
        
        elif len(parameters1) == 3:
            p1 = parameters1[0:1]+parameters1[2:3]
            p2 = parameters2[0:1]+parameters2[2:3]
            if (parameters1[1] == parameters2[1]) and (hasSameElements(p1, p2)):
                return True
            else:
                return False
        
        elif len(parameters1) == 4:
            if (parameters1[0] == parameters2[0]) and (hasSameElements(parameters1[2:4], parameters2[2:4])):
                return True
            else:
                return False
    
    
    if descriptor1 == descriptor2:
        return True
    else:
        return False
        
    #return False #Dummy return statement
    
    

def isPresent(path, descriptor):
    if exists(path):
        with open(path, "r") as f:
            descriptor = descriptor.lower()
            flag1 = False
            for line in f:
                line = line.split("\n")[0].lower()
                if isSame(descriptor, line):
                    flag1 = True
        if flag1:
            return True
        else:
            return False
    else:
        return False

def addFeature(path, descriptor, exercise):
    if isPresent("Exercises/"+str(exercise)+"/EssentialFeatures.csv", descriptor) or isPresent("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv", descriptor):
        print("This feature is already added for this exercise!")
    else:
        with open(path, "a") as f:
            f.write(descriptor+"\n")

def deleteFeature(exercise, descriptor):
    path = ""
    if isPresent("Exercises/"+str(exercise)+"/EssentialFeatures.csv", descriptor):
        path = "Exercises/"+str(exercise)+"/EssentialFeatures.csv"
    elif isPresent("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv", descriptor):
        path = "Exercises/"+str(exercise)+"/NonEssentialFeatures.csv"
    else:
        print("This feature is already deleted for this exercise")
        return
    
    with open("Buffer.csv", "w") as b:
        with open(path, "r") as f:
            for line in f:
                if isSame(descriptor.lower(), line.split("\n")[0].lower()):
                    continue
                b.write(line)
    
    with open("Buffer.csv", "r") as b:
        with open(path, "w") as f:
            for line in b:
                f.write(line)
    
    os.remove("Buffer.csv")

def printFeatures(exercise):
    if exists("Exercises/"+str(exercise)+"/EssentialFeatures.csv"):
        with open("Exercises/"+str(exercise)+"/EssentialFeatures.csv", "r") as ef:
            print("Essential features:")
            for line in ef:
                print(line)
    print("")
    if exists("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv"):
        with open("Exercises/"+str(exercise)+"/NonEssentialFeatures.csv", "r") as nef:
            print("Non-Essential features:")
            for line in nef:
                print(line)
    

exercise = int(input("Enter the exercise number: "))

print("Menu:")
print("1. Add features")
print("2. Delete features")
print("3. List added features")
ch = int(input("Enter choice: "))

if ch == 1:
    flag = True
    while(flag):
        path = ""
        print("")
        ch1 = input("Is the feature essential? Y/N: ")
        if ch1.lower() == 'y':
            path = "Exercises/"+str(exercise)+"/EssentialFeatures.csv"
        else:
            path = "Exercises/"+str(exercise)+"/NonEssentialFeatures.csv"
        
        descriptor = ""
        descriptor += input("Enter the dimension of the feature: ")
        descriptor += ", "
        descriptor += input("Enter the type of the feature: ").upper()
        descriptor += ", "
        descriptor += input("Enter the feature parameters as <, > separated values: ")
        
        addFeature(path, descriptor, exercise)
        
        ch1 = input("Add more features? Y/N: ")
        if ch1.lower() == 'n':
            flag = False
            
elif ch == 2:
    flag = True
    while(flag):
        print("")
        
        descriptor = ""
        descriptor += input("Enter the dimension of the feature: ")
        descriptor += ", "
        descriptor += input("Enter the type of the feature: ").upper()
        descriptor += ", "
        descriptor += input("Enter the feature parameters as <, > separated values: ")
        
        deleteFeature(exercise, descriptor)
        
        ch1 = input("Delete more features? Y/N: ")
        if ch1.lower() == 'n':
            flag = False
            
elif ch == 3:
    print("")
    printFeatures(exercise)
    
else:
    print("Wrong choice!")