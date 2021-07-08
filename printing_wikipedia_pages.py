from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pyautogui as pg
import time

PATH = "C:\webdrivers\chromedriver.exe"
driver = webdriver.Chrome(PATH)

driver.get("https://en.wikipedia.org/wiki/Laptop#:~:text=A%20laptop%20(also%20laptop%20computer,inside%20of%20the%20lower%20lid.")
print(driver.title)

main = driver.find_element_by_id("content")
print(main.text)
