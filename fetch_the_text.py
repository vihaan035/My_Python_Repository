from selenium import webdriver

driver_path = "C:/webdrivers/chromedriver.exe"
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

option = webdriver.ChromeOptions()
option.binary_location = brave_path
# option.add_argument("--incognito") OPTIONAL
# option.add_argument("--headless") OPTIONAL

# Create new Instance of Chrome
driver = webdriver.Chrome(executable_path=driver_path, chrome_options=option)

driver.get("https://www.livecareer.com/resume-search/r/itil-coordinator-incident-manager-920b52d2d59647d89c59e0dbb82c72dd")
main=driver.find_element_by_xpath('//*[@id="document"]')
print(main.text)