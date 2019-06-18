import os
import json
import xlsxwriter
from glob import glob

#set directory to self to load the JSONS
path = "C:\\Users\jebo\Google Drive\AI 2018-2019\Master Project\JSON\Jete"
directory = "."

subdirs = next(os.walk('.'))[1]
print(subdirs)

# Create a workbook and add a worksheet
workbook = xlsxwriter.Workbook('Jete_Keypoints.xlsx')

# Add a bold format to use to highlight cells
bold = workbook.add_format({'bold': True})

#Keypoints contains all the keypoints for pose (1), hand_left (2) and hand_right(3) per Plie
#So first 3 are frame 1, next 3 are frame 2, etc....
keypoints = []
posedata = []
folder = []


for x in range(len(subdirs)):
    os.chdir(subdirs[x])
    print(subdirs[x])
    currentpath = os.getcwd()
    foldername = os.path.basename(currentpath)
    worksheet = workbook.add_worksheet(foldername)
    # Write data headers
    worksheet.write('A1', 'Frame #', bold)
    worksheet.write('B1', 'NoseX', bold)
    worksheet.write('C1', 'NoseY', bold)
    worksheet.write('D1', 'NoseC', bold)
    worksheet.write('E1', 'NeckX', bold)
    worksheet.write('F1', 'NeckY', bold)
    worksheet.write('G1', 'NeckC', bold)
    worksheet.write('H1', 'RShoulderX', bold)
    worksheet.write('I1', 'RShoulderY', bold)
    worksheet.write('J1', 'RShoulderC', bold)
    worksheet.write('K1', 'RElbowX', bold)
    worksheet.write('L1', 'RElbowY', bold)
    worksheet.write('M1', 'RElbowC', bold)
    worksheet.write('N1', 'RWristX', bold)
    worksheet.write('O1', 'RWristY', bold)
    worksheet.write('P1', 'RWristC', bold)
    worksheet.write('Q1', 'LShoulderX', bold)
    worksheet.write('R1', 'LShoulderY', bold)
    worksheet.write('S1', 'LShoulderC', bold)
    worksheet.write('T1', 'LElbowX', bold)
    worksheet.write('U1', 'LElbowY', bold)
    worksheet.write('V1', 'LElbowC', bold)
    worksheet.write('W1', 'LWristX', bold)
    worksheet.write('X1', 'LWristY', bold)
    worksheet.write('Y1', 'LWristC', bold)
    worksheet.write('Z1', 'MidHipX', bold)
    worksheet.write('AA1', 'MidHipY', bold)
    worksheet.write('AB1', 'MidHipC', bold)
    worksheet.write('AC1', 'RHipX', bold)
    worksheet.write('AD1', 'RHipY', bold)
    worksheet.write('AE1', 'RHipC', bold)
    worksheet.write('AF1', 'RKneeX', bold)
    worksheet.write('AG1', 'RKneeY', bold)
    worksheet.write('AH1', 'RKneeC', bold)
    worksheet.write('AI1', 'RAnkleX', bold)
    worksheet.write('AJ1', 'RAnkleY', bold)
    worksheet.write('AK1', 'RAnkleC', bold)
    worksheet.write('AL1', 'LHipX', bold)
    worksheet.write('AM1', 'LHipY', bold)
    worksheet.write('AN1', 'LHipC', bold)
    worksheet.write('AO1', 'LKneeX', bold)
    worksheet.write('AP1', 'LKneeY', bold)
    worksheet.write('AQ1', 'LKneeC', bold)
    worksheet.write('AR1', 'LAnkleX', bold)
    worksheet.write('AS1', 'LAnkleY', bold)
    worksheet.write('AT1', 'LAnkleC', bold)
    worksheet.write('AU1', 'REyeX', bold)
    worksheet.write('AV1', 'REyeY', bold)
    worksheet.write('AW1', 'REyeC', bold)
    worksheet.write('AX1', 'LEyeX', bold)
    worksheet.write('AY1', 'LEyeY', bold)
    worksheet.write('AZ1', 'LEyeC', bold)
    worksheet.write('BA1', 'REarX', bold)
    worksheet.write('BB1', 'REarY', bold)
    worksheet.write('BC1', 'REarC', bold)
    worksheet.write('BD1', 'LEarX', bold)
    worksheet.write('BE1', 'LEarY', bold)
    worksheet.write('BF1', 'LEarC', bold)
    worksheet.write('BG1', 'LBigToeX', bold)
    worksheet.write('BH1', 'LBigToeY', bold)
    worksheet.write('BI1', 'LBigToeC', bold)
    worksheet.write('BJ1', 'LSmallToeX', bold)
    worksheet.write('BK1', 'LSmallToeY', bold)
    worksheet.write('BL1', 'LSmallToeC', bold)
    worksheet.write('BM1', 'LHeelX', bold)
    worksheet.write('BN1', 'LHeelY', bold)
    worksheet.write('BO1', 'LHeelC', bold)
    worksheet.write('BP1', 'RBigToeX', bold)
    worksheet.write('BQ1', 'RBigToeY', bold)
    worksheet.write('BR1', 'RBigToeC', bold)
    worksheet.write('BS1', 'RSmallToeX', bold)
    worksheet.write('BT1', 'RSmallToeY', bold)
    worksheet.write('BU1', 'RSmallToeC', bold)
    worksheet.write('BV1', 'RHeelX', bold)
    worksheet.write('BW1', 'RHeelY', bold)
    worksheet.write('BX1', 'RHeelC', bold)
    #worksheet.write('BY1', 'Class', bold)
    # Start from the first cell below the headers.
    row = 0
    col = 0
    #Set count number for iteration
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            count +=1
            row += 1
            col += 1
            colloop = 1
            with open(filename) as json_data:
                data = json.load(json_data)
                #Create Frame # row in the Excel
                worksheet.write(row,0,count)
                #Pose data contains 75 keypoints (3x25) based on x1,y1 and c1 in order:
                #Nose	Neck	Rshoulder	RElbow	RWrist	Lshoulder	LshouldeL	LElbow	LWList	MidHip	RHip	RKnee	RAnkle	LHip	LKnee	LAnkle	REye	LEye	REar	LEar	LBigToe	LSmallToe	LHeel	RBigToe	RSmallToe	RHeel
                pose = data['people'][0]['pose_keypoints_2d']
                left_hand = data['people'][0]['hand_left_keypoints_2d']
                right_hand = data['people'][0]['hand_right_keypoints_2d']
                keypoints.append(pose)
                posedata.append(pose)
                keypoints.append(left_hand)
                keypoints.append(right_hand)
                #Add data to Excel:
                for bla in range(0,74):
                    worksheet.write_number(row,colloop,pose[bla])
                    colloop += 1
                #worksheet.write(row,colloop,pose[1])
                #colloop += 1
                #worksheet.write(row,colloop,pose[2])
                #colloop += 1
                worksheet.write(row,colloop,pose[3])
                #print("\n Body keypoints - Frame",count,":")
                #print(pose)
                #print("\n Left hand keypoints - Frame",count,":")
                #print(left_hand)
                #print("\n Right hand keypoints - Frame",count,":")
                #print(right_hand)
            continue
        else:
            continue
    os.chdir(path)


#print("\n All keypoints are:")
#print(keypoints)
#print(posedata)

workbook.close()