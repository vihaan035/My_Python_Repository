import pyautogui
from time import sleep

sleep(2)

for i in range(10):
    pyautogui.click(712,600)
    sleep(0.01)
    pyautogui.typewrite('asha')
    pyautogui.press("enter")
    sleep(0.01)
