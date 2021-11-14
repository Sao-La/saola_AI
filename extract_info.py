import pytesseract
import argparse
import cv2
import os
os.environ["TESSDATA_PREFIX"] = "./"
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
    "post_location": "Địa điểm rao bán?",
    "contact": "Số điện thoại người bán?",
}
print("Finish set-up")

def _extract_text(fpath: str):
    img = cv2.imread(fpath)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, lang="vie")
    # correction
    lines = [utils.correct(x) for x in text.split("\n") if len(x.strip()) > 0]
    return " \n ".join(lines)

def _extract_details(context: str):
    global answer_extractor, template_questions
    context = utils.segment(context)
    details = {}
    for qid, question in template_questions.items():
        outputs = answer_extractor(question=question, context=context)
        details[qid] = {"answer": outputs["answer"].strip(), "score": outputs["score"]}
    return details

def extract_info(fpath: str):
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
