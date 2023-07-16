import datetime

today = datetime.date.today()

yesterday = today - datetime.timedelta(days=3)
yesterday = yesterday.strftime("%d/%m/%Y")
print(yesterday)