import math
import numpy as np
import time
import sys

class feature:
    
    def __init__(self, parameters, isEssential, visThreshold):
        self.parameters = parameters
        self.original_parameters = parameters
        self.isEssential = isEssential
        self.visThreshold = visThreshold
        self.keypoints = []
        self.value = ["None"]
        
    def checkVisibility(self):
        allVisible = True
        for parameter in self.parameters:
            if (type(parameter) == int) and (self.keypoints[parameter][4] < self.visThreshold):
                allVisible = False
                break
        if not allVisible:
            self.value = ["None" for x in self.value]
        return allVisible
    
    def normaliseKeypoints(self, id1, id2, keypoints):
        x = (keypoints[id1][1] + keypoints[id2][1])/2
        y = (keypoints[id1][2] + keypoints[id2][2])/2
        z = (keypoints[id1][3] + keypoints[id2][3])/2
        
        visibility = min(keypoints[id1][4], keypoints[id2][4])
        
        id = len(keypoints)
        
        keypoints.append([id, x, y, z, visibility])
        return keypoints
    
    def loadData(self, keypoints):
        self.parameters = []
        i=0
        while(i<len(self.original_parameters)):
            if (type(self.original_parameters[i]) != int) and (self.original_parameters[i].lower() == 'm'):
                self.parameters.append(len(keypoints))
                keypoints = self.normaliseKeypoints(self.original_parameters[i+1], self.original_parameters[i+2], keypoints)
                i = i + 3
            else:
                self.parameters.append(self.original_parameters[i])
                i += 1
        
        self.keypoints = keypoints
        

class distance_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2d'
    
    def calculate(self, video, o_fps):
        if len(self.parameters) != 2 and len(self.parameters) != 4:
            return
        
        if len(self.parameters) == 2:

            first_point = []
            first_point.append(self.keypoints[self.parameters[0]][1])
            first_point.append(self.keypoints[self.parameters[0]][2])

            second_point = []
            second_point.append(self.keypoints[self.parameters[1]][1])
            second_point.append(self.keypoints[self.parameters[1]][2])
        
            distance = math.sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

            self.value = [distance]
        
        elif len(self.parameters) == 4:
            
            first_point = []
            first_point.append(self.keypoints[self.parameters[0]][1])
            first_point.append(self.keypoints[self.parameters[0]][2])

            second_point = []
            second_point.append(self.keypoints[self.parameters[1]][1])
            second_point.append(self.keypoints[self.parameters[1]][2])
        
            distance1 = math.sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)

            third_point = []
            third_point.append(self.keypoints[self.parameters[2]][1])
            third_point.append(self.keypoints[self.parameters[2]][2])

            fourth_point = []
            fourth_point.append(self.keypoints[self.parameters[3]][1])
            fourth_point.append(self.keypoints[self.parameters[3]][2])
        
            distance2 = math.sqrt((third_point[0]-fourth_point[0])**2 + (third_point[1]-fourth_point[1])**2)

            ratio = "None"
            if distance2 != 0:
                ratio = distance1/distance2
            
            self.value = [ratio]
            
        self.checkVisibility()


class keypoint_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2k'

    def calculate(self, video, o_fps):
        if len(self.parameters) != 1:
            return
        
        keypoint = self.keypoints[self.parameters[0]]
        self.value = keypoint[1:3] + keypoint[4:5]
        self.checkVisibility()


class angle_2D(feature):

    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2a'

    def calculate(self, video, o_fps):
        if len(self.parameters) != 4:
            return

        first_point = []
        first_point.append(self.keypoints[self.parameters[0]][1])
        first_point.append(self.keypoints[self.parameters[0]][2])

        second_point = []
        second_point.append(self.keypoints[self.parameters[1]][1])
        second_point.append(self.keypoints[self.parameters[1]][2])

        third_point = []
        
        if type(self.parameters[2]) == int:
            third_point.append(self.keypoints[self.parameters[2]][1])
            third_point.append(self.keypoints[self.parameters[2]][2])

        elif self.parameters[2].lower() == 'x':
            third_point.append(self.keypoints[self.parameters[1]][1] + 1)
            third_point.append(self.keypoints[self.parameters[1]][2])

        elif self.parameters[2].lower() == 'y':
            third_point.append(self.keypoints[self.parameters[1]][1])
            third_point.append(self.keypoints[self.parameters[1]][2] + 1)
        
        A = np.array([(first_point[0]-second_point[0]),(first_point[1]-second_point[1])])
        B = np.array([(third_point[0]-second_point[0]),(third_point[1]-second_point[1])])
        
        modA = np.linalg.norm(A)
        modB = np.linalg.norm(B)
        
        dotProd = np.dot(A, B)
        crossProd = np.cross(A,B)
        modCrossProd = np.linalg.norm(crossProd)
        
        ang = ["None"]
        dir = ["None"]
        if modA == 0 or modB == 0:
            ang = ["None"]
            dir = ["None"]
        else:
            if modCrossProd != 0:
                dir = [crossProd/modCrossProd]
            else:
                dir = ["None"]
               
            cos = dotProd/(modA*modB) 
            if cos >= -1 and cos <= 1:
                ang = [self.toDegree(math.acos(cos))]
            else:
                ang = ["None"]
                
        if self.parameters[3].lower() == 'd':
            self.value = ang + dir
        elif self.parameters[3].lower() == 'nd':
            self.value = ang
            
        self.checkVisibility()
    
    def toDegree(self, ang):
        return ang*(180/math.pi)   
    
class velocity_2D(feature):
    
    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2v'
        self.video = -1
        #self.prevTime = -1
        #self.currTime = -1
        self.prevKeypoints = []
        self.currKeypoints = []
        self.factor = 1
        
    def findAngVel(self, currA, vA, currB, vB):
        rAB = currA - currB
        modrAB = np.linalg.norm(rAB)
        vAB = vA - vB
        modvAB = np.linalg.norm(vAB)
        
        crossProd = np.cross(rAB, vAB)
        modCrossProd = np.linalg.norm(crossProd)
        
        dir = ["None"]
        mag = ["None"]
        if modCrossProd != 0:
            dir = [crossProd/modCrossProd]
        if modrAB != 0:
            mag = [modvAB/modrAB]
        return mag + dir
        
    def calculate(self, video, o_fps): #Currently, unit of time used in calculations is sec
        if len(self.parameters) != 1 and len(self.parameters) != 2 and len(self.parameters) != 3 and len(self.parameters) != 4:
            return
        
        if self.checkVisibility():
            if video != self.video:
                #self.prevTime = time.time()*1000000
                self.prevKeypoints = self.keypoints
                self.video = video
                self.value = ["None", "None"]

            else:
                #self.currTime = time.time()*1000000
                self.currKeypoints = self.keypoints

                #tDiffMili = (self.currTime - self.prevTime)/1000
                if o_fps == 0:
                    print("\nExiting: FPS = 0 for video = "+str(video)+"\n")
                    sys.exit(0)
                    
                tDiffSec = (self.factor/o_fps)
                
                if len(self.parameters) == 1:
                    prevPoint = []
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][1])
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][2])
                    prevPoint = np.array(prevPoint)
                    
                    currPoint = []
                    currPoint.append(self.currKeypoints[self.parameters[0]][1])
                    currPoint.append(self.currKeypoints[self.parameters[0]][2])
                    currPoint = np.array(currPoint)
                    
                    d = currPoint - prevPoint
                    self.value = list(d/tDiffSec)
                    
                elif len(self.parameters) == 2:
                    prevA = []
                    prevA.append(self.prevKeypoints[self.parameters[0]][1])
                    prevA.append(self.prevKeypoints[self.parameters[0]][2])
                    prevA = np.array(prevA)
                    
                    currA = []
                    currA.append(self.currKeypoints[self.parameters[0]][1])
                    currA.append(self.currKeypoints[self.parameters[0]][2])
                    currA = np.array(currA)
                    
                    dA = currA - prevA
                    vA = dA/tDiffSec
                    
                    
                    prevB = []
                    prevB.append(self.prevKeypoints[self.parameters[1]][1])
                    prevB.append(self.prevKeypoints[self.parameters[1]][2])
                    prevB = np.array(prevB)
                    
                    currB = []
                    currB.append(self.currKeypoints[self.parameters[1]][1])
                    currB.append(self.currKeypoints[self.parameters[1]][2])
                    currB = np.array(currB)
                    
                    dB = currB - prevB
                    vB = dB/tDiffSec
                    
                    self.value = self.findAngVel(currA, vA, currB, vB)
                
                elif len(self.parameters) == 3:
                    prevA = []
                    prevA.append(self.prevKeypoints[self.parameters[0]][1])
                    prevA.append(self.prevKeypoints[self.parameters[0]][2])
                    prevA = np.array(prevA)
                    
                    currA = []
                    currA.append(self.currKeypoints[self.parameters[0]][1])
                    currA.append(self.currKeypoints[self.parameters[0]][2])
                    currA = np.array(currA)
                    
                    dA = currA - prevA
                    vA = dA/tDiffSec
                    
                    
                    prevB = []
                    prevB.append(self.prevKeypoints[self.parameters[1]][1])
                    prevB.append(self.prevKeypoints[self.parameters[1]][2])
                    prevB = np.array(prevB)
                    
                    currB = []
                    currB.append(self.currKeypoints[self.parameters[1]][1])
                    currB.append(self.currKeypoints[self.parameters[1]][2])
                    currB = np.array(currB)
                    
                    dB = currB - prevB
                    vB = dB/tDiffSec
                    
                    
                    prevC = []
                    prevC.append(self.prevKeypoints[self.parameters[2]][1])
                    prevC.append(self.prevKeypoints[self.parameters[2]][2])
                    prevC = np.array(prevC)
                    
                    currC = []
                    currC.append(self.currKeypoints[self.parameters[2]][1])
                    currC.append(self.currKeypoints[self.parameters[2]][2])
                    currC = np.array(currC)
                    
                    dC = currC - prevC
                    vC = dC/tDiffSec
                    
                    omega1 = self.findAngVel(currA, vA, currB, vB)
                    omega2 = self.findAngVel(currC, vC, currB, vB)
                    
                    if (omega1[0] == "None") or (omega1[1] == "None") or (omega2[0] == "None") or (omega2[1] == "None"):
                        self.value = ["None", "None"]
                    else:
                        omega1 = omega1[0] * omega1[1]
                        omega2 = omega2[0] * omega2[1]
                        omega = omega1 - omega2
                        
                        dir = ["None"]
                        mag = ["None"]
                        if abs(omega) != 0:
                            dir = [omega/abs(omega)]
                        mag = [abs(omega)]
                        
                        self.value = mag + dir
                    
                elif len(self.parameters) == 4: #Scaled velocity; Sample params: [0, 'r', 1, 2]
                    prevPoint = []
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][1])
                    prevPoint.append(self.prevKeypoints[self.parameters[0]][2])
                    prevPoint = np.array(prevPoint)
                    
                    currPoint = []
                    currPoint.append(self.currKeypoints[self.parameters[0]][1])
                    currPoint.append(self.currKeypoints[self.parameters[0]][2])
                    currPoint = np.array(currPoint)
                    
                    d = currPoint - prevPoint
                    
                    first_point = []
                    first_point.append(self.keypoints[self.parameters[2]][1])
                    first_point.append(self.keypoints[self.parameters[2]][2])
                    first_point = np.array(first_point)
                    
                    second_point = []
                    second_point.append(self.keypoints[self.parameters[3]][1])
                    second_point.append(self.keypoints[self.parameters[3]][2])
                    second_point = np.array(second_point)
                    
                    L = second_point - first_point
                    modL = np.linalg.norm(L)
                    
                    if modL == 0:
                        self.value = ["None", "None"]
                    else:
                        self.value = list(d/(modL * tDiffSec))
                    
                
                #self.prevTime = self.currTime
                self.prevKeypoints = self.currKeypoints
            self.factor = 1
        else:
            self.value = ["None", "None"]
            self.factor += 1
            
class operation_2D(feature):
        
    def add_v(v1, v2):
        if len(v1) == len(v2):
            return list(np.array(v1) + np.array(v2))
        else:
            if len(v1) == 1:
                v1 = [0,0] + v1
            if len(v1) == 2:
                v1 = v1 + [0]
            if len(v2) == 1:
                v2 = [0,0] + v2
            if len(v2) == 2:
                v2 = v2 + [0]
            return list(np.array(v1) + np.array(v2))
        
    def add_nv(s1, s2):
        return [s1[0] + s2[0]]
    
    def add_vnv(v, s):
        return [x+s[0] for x in v]
    
    def sub_v(v1, v2):
        if len(v1) == len(v2):
            return list(np.array(v1) - np.array(v2))
        else:
            if len(v1) == 1:
                v1 = [0,0] + v1
            if len(v1) == 2:
                v1 = v1 + [0]
            if len(v2) == 1:
                v2 = [0,0] + v2
            if len(v2) == 2:
                v2 = v2 + [0]
            return list(np.array(v1) - np.array(v2))
        
    def sub_nv(s1, s2):
        return [s1[0] - s2[0]]
    
    def sub_vnv(v, s):
        return [x-s[0] for x in v]
    
    def mul_v(v1, v2):
        if len(v1) == len(v2):
            if len(v1) == 1:
                return [0]
            if len(v1) == 2:
                return [float(np.cross(np.array(v1), np.array(v2)))]
            if len(v1) == 3:
                return [float(x) for x in np.cross(np.array(v1), np.array(v2))]
        else:
            if len(v1) == 1:
                v1 = [0,0] + v1
            if len(v1) == 2:
                v1 = v1 + [0]
            if len(v2) == 1:
                v2 = [0,0] + v2
            if len(v2) == 2:
                v2 = v2 + [0]
            return [float(x) for x in np.cross(np.array(v1), np.array(v2))]
        
    def mul_nv(s1, s2):
        return [s1[0] * s2[0]]
    
    def mul_vnv(v, s):
        return [x*s[0] for x in v]
    
    def div_nv(s1, s2):
        if s2[0] != 0:
            return [s1[0] / s2[0]]
        else:
            return  ["None"]
    
    def div_vnv(v, s):
        if s[0] != 0:
            return [x/s[0] for x in v]
        else:
            return ["None" for x in v]
    
    def mod_nv(s1, s2):
        if s2[0] != 0:
            return [s1[0] % s2[0]]
        else:
            return  ["None"]
    
    def mod_vnv(v, s):
        if s[0] != 0:
            return [x%s[0] for x in v]
        else:
            return ["None" for x in v]
    
    def __init__(self, parameters, isEssential, visThreshold):
        feature.__init__(self, parameters, isEssential, visThreshold)
        self.type = '2opt'
        self.isValid = False
        
        i = 0
        indices1 = []
        indices2 = []
        for o in operators:
            indices2 = [x for x, j in enumerate([x.split('_')[0] if type(x) == str else x for x in self.original_parameters]) if j == o]
            indices1 = list(set(indices1).union(set(indices2)))
        indices1.sort()
        
        opt_indices = [x for x, j in enumerate(self.original_parameters) if j == 'opt']
        opt_indices.sort()
        
        i_type = {}
        for index in indices1:
            i_type[index] = 'c'
        for index in opt_indices:
            i_type[index] = 'o'
        
        count = 0
        self.operator = ''
        all_indices = list(set(indices1).union(set(opt_indices)))
        all_indices.sort()
        for index in all_indices:
            if i_type[index] == 'o':
                count += 1
            else:
                count -= 1
            if count == -1:
                i = index
                self.operator = self.original_parameters[i]
                break
        
        descriptor1 = self.original_parameters[0:i]
        descriptor2 = self.original_parameters[(i+1):len(self.original_parameters)]
        
        featureType1 = str(descriptor1[0]) + descriptor1[1]
        featureType2 = str(descriptor2[0]) + descriptor2[1]
        
        self.feature1 = object_dispatcher[featureType1](descriptor1[2:len(descriptor1)], isEssential, visThreshold)
        self.feature2 = object_dispatcher[featureType2](descriptor2[2:len(descriptor2)], isEssential, visThreshold)
        
        feature1_type = self.feature1.type
        feature2_type = self.feature2.type
        
        if (feature1_type == '2opt') or (feature2_type == '2opt'):
            self.isValid = False
        else:
            subtype = {
                1: 'v',
                4: 'v',
                2: 'o',
                3: 'o'
            }
            if feature1_type[1:] == 'v':
                feature1_type += subtype[len(descriptor1) - 2 - descriptor1.count('m')*2]
            if feature2_type[1:] == 'v':
                feature2_type += subtype[len(descriptor2) - 2 - descriptor2.count('m')*2]
            
            if feature1_type[1:] == 'a':
                feature1_type += self.original_parameters[i-1]
            if feature2_type[1:] == 'a':
                feature2_type += self.original_parameters[-1]
                
            if feature1_type[0:4] == '2opt':
                feature1_type = feature1_type[0] + feature1_type.split('opt')[-1].split('_')[1]
            if feature2_type[0:4] == '2opt':
                feature2_type = feature2_type[0] + feature2_type.split('opt')[-1].split('_')[1]
                
            if (feature1_type[0] == '2') and (feature2_type[0] == '2') and (feature1_type == feature2_type):
                if feature1_type[1:] in validOperands[self.operator.split('_')[0]]:
                    self.type += ('_' + feature1_type[1:])
                    self.isValid = True
        
    def loadData(self, keypoints):
        self.keypoints = keypoints
        self.feature1.loadData(self.keypoints)
        self.feature2.loadData(self.keypoints)
    
    def calculate(self, video, o_fps):
        if self.isValid == False:
            return
        
        self.feature1.calculate(video, o_fps)
        self.feature2.calculate(video, o_fps)
        
        value1 = self.feature1.value
        value2 = self.feature2.value
        
        if ("None" in value1) or ("None" in value2):
            return
        
        op_type = self.type.split('_')[1]
        if '_' in self.operator:
            if op_type in ['d', 'and', 'vv']:
                self.value = vnv_func[self.operator.split('_')[0]](value1, [float(self.operator.split('_')[1])])
            elif op_type in ['ad', 'vo']:
                v = [value1[0]*value1[1]]
                res = vnv_func[self.operator.split('_')[0]](v, [float(self.operator.split('_')[1])])[0]
                if res == 0:
                    self.value = [res, "None"]
                else:
                    self.value = [abs(res), res/abs(res)]
            elif op_type == 'k':
                v = value1[0:-1]
                res = vnv_func[self.operator.split('_')[0]](v, [float(self.operator.split('_')[1])])
                self.value = res + [value1[-1]]
        else:
            if op_type in ['d', 'and']:
                self.value = nv_func[self.operator](value1, value2)
            elif op_type in ['ad', 'vo']:
                v1 = [value1[0]*value1[1]]
                v2 = [value2[0]*value2[1]]
                res = v_func[self.operator](v1, v2)[0]
                if res == 0:
                    self.value = [res, "None"]
                else:
                    self.value = [abs(res), res/abs(res)]
            elif op_type == 'k':
                v1 = value1[0:-1]
                v2 = value2[0:-1]
                res = v_func[self.operator](v1, v2)
                self.value = res + [min(value1[-1], value2[-1])]
            elif op_type == 'vv':
                self.value = v_func[self.operator](value1, value2)
    
object_dispatcher = {
    '2d': distance_2D,
    '2k': keypoint_2D,
    '2a': angle_2D,
    '2v': velocity_2D,
    '2opt': operation_2D
    #'3d': distance_3D,
    #'3k': keypoint_3D,
    #'3a': angle_3D,
    #'3v': velocity_3D,
    #'3opt': operation_3D
}

operators = ['add', 'sub', 'mul', 'div', 'mod']

validOperands = {
    'add': ['d', 'k', 'ad', 'and', 'vv', 'vo'],
    'sub': ['d', 'k', 'ad', 'and', 'vv', 'vo'],
    'mul': ['d', 'k', 'ad', 'and', 'vv', 'vo'],
    'div': ['d', 'and'],
    'mod': ['d', 'and']
}

v_func = {
    'add': operation_2D.add_v,
    'sub': operation_2D.sub_v,
    'mul': operation_2D.mul_v
}

nv_func = {
    'add': operation_2D.add_nv,
    'sub': operation_2D.sub_nv,
    'mul': operation_2D.mul_nv,
    'div': operation_2D.div_nv,
    'mod': operation_2D.mod_nv
}

vnv_func = {
    'add': operation_2D.add_vnv, 
    'sub': operation_2D.sub_vnv,
    'mul': operation_2D.mul_vnv,
    'div': operation_2D.div_vnv,
    'mod': operation_2D.mod_vnv
}