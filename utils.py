import difflib
import pylcs
import phonenumbers
import dateparser
from vncorenlp import VnCoreNLP

segmenter = VnCoreNLP("./resources/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
words = [w.strip().lower() for w in open("./resources/vi-DauMoi.dic", "r")]
animal_names = [w.strip().lower() for w in open("./resources/vi-animals.txt", "r")]

def correct(text: str, threshold: float = 0.8):
    def normalize(token: str):
        return "".join(e for e in token if e.isalnum()).lower()
    def compute_dist(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()
    
    res = []
    for token in text.split():
        norm_token = normalize(token)
        dist = [(compute_dist(norm_token, w), w) for w in words]
        match_ratio, match_token = max(dist)
        if match_ratio >= threshold:
            res.append(match_token)
        else:
            res.append(token)
            
    return " ".join(res)

def segment(text: str):
    global segmenter
    res = []
    for sent in segmenter.tokenize(text):
        res.append(" ".join(sent))
    return " \n ".join(res)

def unsegment(text: str):
    return " ".join(text.split("_"))

def match_animal(text: str):
    res = []
    for x in animal_names:
        match_len = pylcs.lcs2(text, x)
        match_ratio = match_len / len(x)
        res.append((match_ratio, x))
    match_name = max(res)[1]
    return match_name

def parse_phone(text: str):
    res = []
    for match in phonenumbers.PhoneNumberMatcher(text, "VN"):
        res.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))
    return res

def parse_date(text: str):
    date = dateparser.parse(
        text, languages=["vi", "fr", "en"], 
        settings={"TIMEZONE": "Asia/Ho_Chi_Minh", "PREFER_DAY_OF_MONTH": "first"},
    )
    return "" if not date else date.strftime("%d/%m/%Y")
