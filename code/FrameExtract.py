import cv2
import os
from os import listdir
from os.path import isfile, join

files = [f for f in listdir(".\Fondu") if isfile(join(".\Fondu", f))]
print(files)
directory = "."
slash = "\\"

os.chdir(".\Fondu")

for i in files:
    vidcap = cv2.VideoCapture(i)
    succes,image = vidcap.read()
    succes = True
    count = 1;
    file = i[0:-4]
    path = directory + slash + file
    os.mkdir(file)
    os.chdir(path)
    while succes:
        framenumber = ("_frame%d.jpg" % count)
        filename = file + framenumber
        print(filename)
        cv2.imwrite(filename, image)
        succes,image = vidcap.read()
        count += 1
    os.chdir("..")