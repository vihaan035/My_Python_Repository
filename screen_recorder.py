import pyautogui
import numpy as np
import cv2

screenShot = pyautogui.screenshot()
screenShot.save("temp.png")
imageResolution = screenShot.size

fourCC = cv2.VideoWriter_fourcc(*"mp4v")
frameRate = 30.0
videoWriter = cv2.VideoWriter("video.mp4", fourCC, frameRate, imageResolution)

for i in range(100):
    screenShot = pyautogui.screenshot()
    numpyImageArray = np.array(screenShot)
    numpyImageArray = cv2.cvtColor(numpyImageArray, cv2.COLOR_BGR2RGB)
    videoWriter.write(numpyImageArray)

cv2.destroyAllWindows()
videoWriter.release()