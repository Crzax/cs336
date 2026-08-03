import re

EMAIL_RE =  re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
IP_RE = re.compile(r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)")

def mask_emails(text: str):
    new_text, count = EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)
    return new_text, count

def mask_phone_numbers(text: str):
    new_text, count = PHONE_RE.subn("|||PHONE_NUMBER|||", text)
    return new_text, count

def mask_ips(text: str):
    new_text, count = IP_RE.subn("|||IP_ADDRESS|||", text)
    return new_text, count