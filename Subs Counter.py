import urllib.request
import json

name=input("Enter username: ")
key = "AIzaSyCrRBrVhVp6Ev58P1z_tf24-REmFmp27uk"

data = urllib.request.urlopen("https://www.googleapis.com/youtube/v3/channels?part=statistics&forUsername="+name+"&key="+key).read()
subs = json.loads(data)["items"][0]["statistics"]["subscriberCount"]

print(name + " has " + "{:,d}".format(int(subs)) + " subscribers!🎉")
