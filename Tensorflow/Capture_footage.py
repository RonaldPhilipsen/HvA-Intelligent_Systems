import cv2 as cv
import os 


cam = cv.VideoCapture(0)


nPictures = 300
i = 0
img_folder = "images"

def set_folder():
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    folder = input("Please enter a folder name: ")
    if not os.path.exists(img_folder + folder):
        os.mkdir(img_folder + '/' + folder)
    return img_folder + '/' + folder + '/'
    


print("This script grabs " + str(nPictures) + " images from your webcam  ")
print("\nUsage:")
print("===================")
print("ESC\t Exits the program")
print("N\t sets new path")
print("Space\t start capture\n\n")

capture = False


path = set_folder()

while True:
    ret, img = cam.read()

    if(ret == False):
        print("Failed to open webcam")
        break
    cv.imshow('Webcam', img)

    key = cv.waitKey(1)
    if key == 27: # esc to quit
       break
    elif key == 32: # space to start/stop capture;
       i = 0
       capture = True
       print("Capture started")
    elif key == ord('n'):
       path = set_folder()

    if(capture):
        cv.imwrite(path + str(i) + ".jpg", img)
        i += 1
        if(i == nPictures):
            print("Capture done")
            capture = False

    
cv.destroyAllWindows()