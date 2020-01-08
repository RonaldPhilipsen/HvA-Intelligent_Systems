#!/usr/bin/python
from PIL import Image
import os, sys

path = ".\\images\\"
dirs = os.listdir( path )

def resize(height, width):
    for dir in dirs:
        for file in os.listdir(path + dir):
            if os.path.isfile(path+ dir + "\\" + file):
                im = Image.open(path+ dir+ "\\" + file)
                imResize = im.resize((height,width), Image.ANTIALIAS)
                imResize.save(path+ dir + "\\" + file, 'JPEG', quality=90)

resize(150,150)