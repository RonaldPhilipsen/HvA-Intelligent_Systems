#!/usr/bin/python
from PIL import Image
import os, sys

path = ".\\data\\"
dirs = os.listdir( path )

def resize(height, width):
    for dir in dirs:
        for file in os.listdir(path + dir):
            if os.path.isfile(path+ dir + "\\" + file):
                im = Image.open(path+ dir+ "\\" + file)
                imResize = im.resize((height,width), Image.ANTIALIAS)
                imResize.save(path+ dir + "\\" + file, 'JPEG', quality=90)

resize(120,120)

#copy files from fingers folder

path = ".\\fingers\\fingers\\train"
outpath = ".\\data\\"
for f in  os.listdir( path ):
     if os.path.isfile(path + "\\" + f):
         fnlen = len(f)
         nfingers = (f[fnlen-6: fnlen-5])
         os.rename(path + "\\" + f, outpath + "\\" + str(nfingers) + "\\" + f )