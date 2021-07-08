import random
answers = ['yah nishchit hai', 'nishchit roop se aisa', 'bina kisee sanshay ke', 'haan bilakul', 'aap us par bharosa kar sakate hain', 'jaisa dikhata hai haan', 'sabase adhik sambhaavana', 'aautaluk achchha hai', 'haan sanket haan kee or ishaara karate hain', 'uttar dhundhalee', 'punah prayaas karen', 'baad mein phir se poochhana', 'achchha hoga ki tumhen abhee nahin bataaya jae', 'ab bhavishyavaanee nahin kar sakate','dhyaan lagao aur phir se poochho','us par bharosa mat karo', 'mera javaab hai nahin', 'mere srot ne mana kiya', 'aautaluk itana achchha nahin hai', 'bahut sandehajanak']


print('  __  __          _____ _____ _____    ___  ')
print(' |  \/  |   /\   / ____|_   _/ ____|  / _ \ ')
print(' | \  / |  /  \ | |  __  | || |      | (_) |')
print(' | |\/| | / /\ \| | |_ | | || |       > _ < ')
print(' | |  | |/ ____ \ |__| |_| || |____  | (_) |')
print(' |_|  |_/_/    \_\_____|_____\_____|  \___/ ')
print('')
print('')
print('')
print('Tumhari kya problem hai baccha?')


def Magic8Ball():
    print('Savaal Puchho')
    input()
    print (answers[random.randint(0, len(answers)-1)] )
    print('Javaab Lo..')
    Replay()
    

def Replay():
    print ('Aur koi samasya? ')
    reply = input()
    if reply == 'Y':
        Magic8Ball()
    elif reply == 'N':
        print("Techno Gamerz ka video dekh lo agar aaya ho to")
        exit()
    else:
        print('Maaf Karna')
        Replay()

		
Magic8Ball()
