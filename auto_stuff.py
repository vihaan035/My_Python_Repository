import pyautogui as pg
import time
import tkinter as tk
root=tk.Tk()
canvas=tk.Canvas(root,width=300,height=300)
canvas.pack()
pg.hotkey('win','s')
time.sleep(1)
pg.typewrite('Google Chrome')
pg.press('enter')
time.sleep(1)
pg.typewrite('youtube.com')
pg.press('enter')
time.sleep(2)
pg.click(x=222, y=210)

def takeScreenShot():
    my=pg.screenshot()
    my.save('screen2.png')
    print("Saved")
    
canvas.quit()
root.mainloop()
