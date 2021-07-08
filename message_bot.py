from twilio.rest import Client 
 
account_sid = 'AC140930447de32849d9cd600d185537e0' 
auth_token = '[AuthToken]' 
client = Client(account_sid, auth_token) 
 
message = client.messages.create( 
                              from_='+16505164495',  
                              body='2 Samose lana bhaiya!!!',      
                              to='+918574073784' 
                          ) 
 
print(message.sid)
