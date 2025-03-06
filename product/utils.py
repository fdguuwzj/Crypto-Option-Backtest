# -*- coding:utf-8 -*-
"""
@FileName：utils.py
@Description：
@Author：fdguuwzj
@Time：2025-03-06 13:16
"""


class MessageSender:
    def __init__(self, key, token, platform='Telegram'):
        self.key = key
        self.token = token
        self.platform = platform


    def send(self, message):
        if self.platform == 'Telegram':
            self._send_telegram_message(message)

        else:
            raise ValueError(f'Unsupported platform: {self.platform}')


    def _send_telegram_message(self, message):
        import requests
        url = f'https://api.telegram.org/bot{self.token}/sendMessage'
        data = {'chat_id': self.key, 'text': message}
        response = requests.post(url, data=data)
        return response.json()

    def _send_slack_message(self, chat_id, message):
        import requests
        url = 'https://slack.com/api/chat.postMessage'
        headers = {'Authorization': f'Bearer {self.token}'}
        data = {'channel': chat_id, 'text': message}
        response = requests.post(url, headers=headers, data=data)
        return response.json()



if __name__ == '__main__':
    ms = MessageSender(key='5426883973', token='7586997310:AAGKmJP2v3w5NGJNfuAECU0aeF64F9LeRRk', platform='Telegram')
    ms.send(message='Hello, this is a 11test message!')