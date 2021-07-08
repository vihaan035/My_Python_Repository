import webbrowser
import tkinter as tk


def onclick1():
    webbrowser.open('https://www.youtube.com/channel/UCI86prlqXhbkREDMTaORvLQ')
    print("Beast Boy Shub")
def onclick2():
    webbrowser.open('https://www.youtube.com/channel/UCIabPXjvT5BVTxRDPCBBOOQ')
    print("Dani")
def onclick3():
    webbrowser.open('https://www.youtube.com/channel/UCzmtKpQQyZ-9ZNY_bbFtVHw')
    print("Dani2")
def onclick4():
    webbrowser.open('https://mail.google.com/mail/u/1/#inbox')
    print("Gmail")



root = tk.Tk()
root.title("Button")

#First step: Create Element
btn1 = tk.Button(root, text="Beast Boy Shub",command=onclick1)
btn2 = tk.Button(root, text="Dani",command=onclick2)
btn3 = tk.Button(root, text="Dani2",command=onclick3)
btn4 = tk.Button(root, text="Gmail",command=onclick4)


#Last step: Put element on the main window
btn1.pack()
btn2.pack()
btn3.pack()
btn4.pack()

root.mainloop()
