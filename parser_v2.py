#!/usr/bin/python
import pathlib
import os
import re
import json
import zipfile
import chardet
from collections import defaultdict

base_path = pathlib.Path(__file__).parent.resolve()
print(base_path)
dataset = os.path.join(base_path, "dataset","raw")
content = defaultdict(set)

for root, dirs, files in os.walk(os.path.join(base_path, "dataset","raw")):
    if not files or "__MACOSX" in root:
        continue
    filename = ".".join(re.split(r"\.|_",root.split("\\")[-1]))
    for f in files:
        if f not in content[filename] and (f.endswith("json") or f.endswith("docx")):
            content[filename].add(os.path.join(root,f))

train = []
test = []
for name, files in content.items():
    for f in files:
        with open(f,"r",encoding="utf-8-sig") as fh:
            content = json.load(fh)
            if "gpt3" in name:
                train.extend(content)
            else:
                test.extend(content)
res = os.path.join(base_path,"dataset","result")
train_file = os.path.join(res,"train.json")
test_file = os.path.join(res,"test.json")
os.makedirs(res,exist_ok=True)
with open(train_file,'w') as fh:
    json.dump(train, fh)
with open(test_file,'w') as fh:
    json.dump(test, fh)
