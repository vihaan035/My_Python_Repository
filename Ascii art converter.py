from PIL import Image
import sys

def convertImgtoAscii(image):
    ascii = ""
    width, height = image.size
    scale = 1
    widen = 2
    #convert image to rgb mapping
    rgb_img = image.convert('RGB')
    #loop through rgb values and translate to ascii values based on thier average        
    for y in range(0, height,scale):
        for x in range(0,width,scale):
            r, g, b = rgb_img.getpixel((x,y))
            avg = (r+g+b)/3
            if avg > 225:
                ascii+=" "*widen
            elif avg > 200:
                ascii+="."*widen
            elif avg > 175:
                ascii+=":"*widen
            elif avg > 150:
                ascii+="-"*widen
            elif avg > 125:
                ascii+="="*widen
            elif avg > 100:
                ascii+="+"*widen
            elif avg > 75:
                ascii+="*"*widen
            elif avg > 50:
                ascii+="#"*widen
            elif avg > 25:
                ascii+="%"*widen
            elif avg > 0:
                ascii+="@"*widen         
        ascii+="\n"
    return ascii
    
def main():
    #Capture file name from standard input
    filename=input("What is the name of the file you like to convert? ")
    #Load Image
    try:
        image = Image.open(filename)
        image.load()
    except IOError:
        print("FILE NOT FOUND")
        quit()
    #Convert image to ascii    
    ascii=convertImgtoAscii(image)
    #Write ascii value to text file 
    try:    
        with open((filename[:filename.find(".")] + ".txt"), "w") as ascii_file:
            ascii_file.write("%s" % ascii)
    except IOError:
        print("COULD NOT WRITE TO FILE")
        quit()
    #print ascii
    print(ascii)
if __name__ == "__main__":
    main()