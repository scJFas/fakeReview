import requests

URL = 'https://sc.ftqq.com/SCU91150T581824bba843050222840524161ac93c5e7b5deb9ad08.send'

def sendMessage(title = '程序运行完成', context = '完成任务'):
    pa = {"text":title, "desp":context}
    requests.get(URL,params=pa)

def main():
    sendMessage()


if __name__ == "__main__":
    main()