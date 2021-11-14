import pytesseract
import argparse
import cv2
import os
import numpy as np
import urllib.request
os.environ["TESSDATA_PREFIX"] = "./resources/"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import pipeline
import utils

answer_extractor = pipeline(
    "question-answering", 
    # model="nguyenvulebinh/vi-mrc-base",
    model="ancs21/xlm-roberta-large-vi-qa",
)
template_questions = {
    "animal_name": "Động vật nào được rao bán?",
    "product_name": "Sản phẩm được rao bán là gì?",
    "usage": "Mục đích sử dụng?",
    "quantity": "Số lượng sản phẩm được rao bán?",
    "post_date": "Thời gian rao bán?",
    "location": "Địa điểm rao bán?",
    "phone": "Số điện thoại người bán?",
}
print("Finish set-up")

def url_to_image(url):
    with urllib.request.urlopen(url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

def _extract_text_with_url(url: str):
    img = url_to_image(url)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang="vie")
    # correction
    lines = [utils.correct(x) for x in text.split("\n") if len(x.strip()) > 0]
    return " \n ".join(lines)


def _extract_text(fpath: str):
    img = cv2.imread(fpath)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang="vie")
    # correction
    lines = [utils.correct(x) for x in text.split("\n") if len(x.strip()) > 0]
    return " \n ".join(lines)

def _extract_details(context: str):
    global answer_extractor, template_questions
    context = utils.segment(context)
    if context == '':
        return {}
    details = {}
    for qid, question in template_questions.items():
        outputs = answer_extractor(question=question, context=context)
        details[qid] = {"answer": utils.unsegment(outputs["answer"].strip()), "score": outputs["score"]}
        if qid == "animal_name":
            details[qid]["answer"] = utils.match_animal(details[qid]["answer"])
        elif qid == "phone":
            details[qid]["answer"] = utils.parse_phone(details[qid]["answer"])
    return details

def extract_info(fpath: str):
    assert os.path.exists(fpath), f"File doesn't exist: {fpath}"
    context = _extract_text(fpath)
    details = _extract_details(context)
    return details

def extract_info_with_url(url: str):
    context = _extract_text_with_url(url)
    details = _extract_details(context)
    return details

if __name__ == "__main__":
    while True:
        url = input()
        details = extract_info_with_url(url)
        print(details)

    # For testing in development phase
    # import json
    # data_dir = "./data"
    # save_dir = "./output"
    # os.makedirs(save_dir, exist_ok=True)
    # for filename in os.listdir(data_dir):
    #     fpath = os.path.join(data_dir, filename)
    #     if not os.path.isfile(fpath):
    #         continue
    #     print("Running with", fpath)
    #     basename = os.path.splitext(filename)[0]
    #     context = _extract_text(fpath)
    #     with open(os.path.join(save_dir, basename + "_text.txt"), "w") as f:
    #         f.write(context)
    #     details = _extract_details(context)
    #     with open(os.path.join(save_dir, basename + "_pred.json"), "w") as f:
    #         f.write(json.dumps(details))
