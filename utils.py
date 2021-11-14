import difflib
from vncorenlp import VnCoreNLP

segmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 
words = [w.strip() for w in open("./vi-DauMoi.dic", "r")]

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