import os

input_dir = 'ModifiedData2'
output_dir = ''
video = 1

l = os.listdir(input_dir)
l.sort()

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
