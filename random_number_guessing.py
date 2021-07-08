import random
number=random.randrange(0,20)
chances=0
guessCheck="wrong"
default_coins=1000
print("Welcome to Number Guess")
print("You have 1000 coins and you have to pay 100 coins to play...")
print("Every chance you miss you will lose 100 coins!")
first_payment=str(input("Do you want to pay 100 coins to someone whom I also don't know? y/n : "))
if first_payment == "y":
        default_coins-=100
        print("You have payed 100 coins to the game!!!")
else:
        print("Ok... Leaving Game")
        quit()

if default_coins<=0:
        print("you don't have coins... you may have to leave")
        quit()
else: 
        while guessCheck=="wrong":
                response=int(input("Please input a number between 0 and 20:"))
                try:
                        val=int(response)
                except ValueError:
                        print("This is not a valid integer. Please try again")
                        continue
                val=int (response)
                if val<number:
                        print("This is lower than actual number. Please try again.")
                        default_coins-=100
                elif val>number:
                        print("This is higher than actual number. Please try again.")
                        default_coins-=100
                else:
                        print("This is the correct number")
                        guessCheck="correct"
print("Thank you for playing Number Guess. See you again","\n","You have done in:",chances,"chances")
