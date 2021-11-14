import pytesseract
import argparse
import cv2
import os
os.environ["TESSDATA_PREFIX"] = "./"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
from transformers import pipeline

answer_extractor = pipeline(
    "question-answering", 
    model="ancs21/xlm-roberta-large-vi-qa",
)
template_questions = {
    "animal_name": "Tên động vật được rao bán?",
    "product_name": "Sản phẩm được rao bán là gì?",
    "quantity": "Bao nhiêu sản phẩm được rao bán?",
    "post_date": "Bài đăng vào thời gian nào?",
    "post_location": "Địa điểm tổ chức rao bán?",
    "contact": "Số điện thoại người bán?",
}

def _extract_text(fpath):
    img = cv2.imread(fpath)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang='vie')
    return text

def _extract_details(context):
    global answer_extractor, template_questions
    details = {}
    for qid, question in template_questions.items():
        outputs = answer_extractor(question=question, context=context)
        details[qid] = {"answer": outputs["answer"].strip(), "score": outputs["score"]}
    return details

def extract_info(fpath):
    assert os.path.exists(fpath), f"File doesn't exist: {fpath}"
    context = _extract_text(fpath)
    details = _extract_details(context)
    return details

if __name__ == "__main__":
    # For testing in development phase
    import json
    data_dir = "./data"
    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(data_dir):
        fpath = os.path.join(data_dir, filename)
        if not os.path.isfile(fpath):
            continue
        print("Running with", fpath)
        basename = os.path.splitext(filename)[0]
        context = _extract_text(fpath)
        with open(os.path.join(save_dir, basename + "_text.txt"), "w") as f:
            f.write(context)
        details = _extract_details(context)
        with open(os.path.join(save_dir, basename + "_pred.json"), "w") as f:
            f.write(json.dumps(details))
