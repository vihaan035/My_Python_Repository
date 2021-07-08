import keyboard

k1=keyboard.record(until="Enter")
print("Enter")

k2=keyboard.record(until="Tab")
print("Tab")

k3=keyboard.record(until="Alt")
print("Alt")

k4=keyboard.record(until="Backspace")
print("Backspace")

keyboard.play(k1,speed_factor=1)
keyboard.play(k2,speed_factor=1)
keyboard.play(k3,speed_factor=1)
keyboard.play(k4,speed_factor=1)