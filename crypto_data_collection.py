from selenium import webdriver

driver_path = "C:/webdrivers/chromedriver.exe"
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

option = webdriver.ChromeOptions()
option.binary_location = brave_path
# option.add_argument("--incognito") OPTIONAL
# option.add_argument("--headless") OPTIONAL

# Create new Instance of Chrome
driver = webdriver.Chrome(executable_path=driver_path, chrome_options=option)

driver.get("https://coinmarketcap.com/")

BTC_price = driver.find_element_by_xpath('//*[@id="__next"]/div/div[2]/div/div/div[2]/table/tbody/tr[1]/td[4]/div/a')
BTC_24h = driver.find_element_by_xpath('//*[@id="__next"]/div/div[2]/div/div/div[2]/table/tbody/tr[1]/td[5]/span')
BTC_7d = driver.find_element_by_xpath('//*[@id="__next"]/div/div[2]/div/div/div[2]/table/tbody/tr[1]/td[6]/span')
BTC_MC = driver.find_element_by_xpath('//*[@id="__next"]/div/div[2]/div/div/div[2]/table/tbody/tr[1]/td[7]/p')
print("Bitcoin Price",BTC_price.text)
print("Bitcoin 24H Price",BTC_24h.text)
print("Bitcoin 7D",BTC_7d.text)
print("Bitcoin",BTC_MC.text)