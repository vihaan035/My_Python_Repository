from selenium import webdriver
import pyautogui as pg
import time
import pyttsx3

main=0
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[len(voices)-1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

driver_path = "C:/webdrivers/chromedriver.exe"
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

option = webdriver.ChromeOptions()
option.binary_location = brave_path
# option.add_argument("--incognito") OPTIONAL
# option.add_argument("--headless") OPTIONAL

# Create new Instance of Chrome
driver = webdriver.Chrome(executable_path=driver_path, chrome_options=option)

driver.get("https://zeenews.india.com/")
print(driver.title)
pg.hotkey('win','down')
main = driver.find_element_by_id("block-views-be8ee82f5a0016aa6bc5a52c9d9bc9b7")
speak(main.text)