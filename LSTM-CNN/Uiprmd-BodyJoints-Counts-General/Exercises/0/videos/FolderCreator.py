import os
import functools

input_dir = 'Exercises\\0\Videos\ModifiedData'
output_dir = 'Exercises\\0\Videos'
video = 1

def cmp(a, b):
    l1 = a[:-4].split("_")
    l2 = b[:-4].split("_")
    if len(l1)>len(l2):
        return 1
    elif len(l1)<len(l2):
        return -1
    else:
        if(int(l1[0][1:])>int(l2[0][1:])):
            return 1
        elif(int(l1[0][1:])<int(l2[0][1:])):
            return -1
        else:
            if(int(l1[1][1:])>int(l1[1][1:])):
                return -1
            else:
                return 1

l = os.listdir(input_dir)
l = sorted(l,key=functools.cmp_to_key(cmp))

for filename in l:
    f = os.path.join(input_dir, filename)

    if os.path.isfile(f):
        output_file_folder = "V"+str(video)
        os.mkdir(os.path.join(output_dir,output_file_folder))

        f1= open(f,"r")
        f2 = open(os.path.join(output_dir,os.path.join(output_file_folder,"Keypoints.txt")),"w")

        data = f1.readlines()
        f2.writelines(data)
        
        f1.close()
        f2.close()

        video += 1
