#!/usr/bin/python
from PIL import Image
import os, sys



def resize(path, height, width):
    dirs = os.listdir( path )
    for dir in dirs:
        for file in os.listdir(path + dir):
            if os.path.isfile(path+ dir + "\\" + file):
                im = Image.open(path+ dir+ "\\" + file)
                imResize = im.resize((width,height), Image.ANTIALIAS)
                imResize.save(path+ dir + "\\" + file, 'JPEG', quality=90)

resize(".\\data\\test\\", 240, 320)
resize(".\\data\\train\\", 240, 320)

#copy files from fingers folder
path = ".\\fingers\\fingers\\train"
outpath = ".\\data\\"
for f in  os.listdir( path ):
     if os.path.isfile(path + "\\" + f):
         fnlen = len(f)
         nfingers = (f[fnlen-6: fnlen-5])
         os.rename(path + "\\" + f, outpath + "\\" + str(nfingers) + "\\" + f )