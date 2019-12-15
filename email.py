#dec 15 2019
import smtplib
import time
start_time = time.time()
s=smtplib.SMTP('smtp.gmail.com',587)
s.starttls()
s.login("jyothiv.1420@gmail.com","*******************")
recv=str(input("Enter the email id of the receiver: "))
message=str(input("Enter the message: "))
s.sendmail("jyothiv.1420@gmail.com",recv,message)
s.quit()
print("The email has been sent")
print("%s seconds" % (time.time() - start_time))
