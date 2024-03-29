import requests

telegram_token = "6645018983:AAG2nTpOuCxwdgfMZTlxkmlBxPchFrm8fec"
chat_id = "903737895"

def send_telegram(text):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=data)
