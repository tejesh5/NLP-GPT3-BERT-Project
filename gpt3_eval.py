import json
import openai
import time
from sklearn.metrics import classification_report, confusion_matrix

openai.api_key = "YOUR KEY HERE"

prompt = """
    Given a set of premises and a hypothesis, label the hypothesis as True, False, or Undetermined.
    Premises: {premise}
    Hypothesis: {hypothesis}
    Label: 
"""

f = open('dataset/result/test.json')
test = json.load(f)

f = open('dataset/result/train.json')
train = json.load(f)

data = test + train

print(len(data))
truth = []
pred = []
cnt = 0
for example in data[3711:]:
    premise = '\n'.join(example['Premise'])
    hypothesis = example['Hypothesis'][0]
    cnt += 1
    response = openai.Completion.create(
        model="text-davinci-003", 
        prompt=prompt.format(premise=premise, hypothesis=hypothesis),
        temperature=0, max_tokens=7)
    label = example['Label'][0].replace('\'', '')
    label = label.replace('\"', '')
    truth.append(label)
    pred.append(response['choices'][0]['text'].strip())
    print(cnt)
    conf_mat = confusion_matrix(truth, pred, labels=['True', 'False', 'Undetermined'])
    print(conf_mat)

conf_mat = confusion_matrix(truth, pred, labels=['True', 'False', 'Undetermined'])

f = open('gpt3_conf_mat_3000-end.json', 'x')
f.write(str(conf_mat))