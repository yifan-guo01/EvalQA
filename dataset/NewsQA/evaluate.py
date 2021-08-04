import collections
import json
import os
import re
import string
import sys



def ropen(f) :
  return open(f,'r',encoding='utf8')

  
def jload(infile) :
  ''' loads .json file, preprocessed from a .txt file'''
  with ropen(infile) as f:
    res = json.load(f)
    return res
    

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1



def make_eval_dict(exact_scores, f1_scores):
  score =dict() 
  total = len(exact_scores)
  score['total'] = total
  score['f1'] = 100.0 * sum(f1_scores.values()) / total
  score['exact'] = 100.0 * sum(exact_scores.values()) / total
  return score


def get_raw_scores(alg):
  exact_scores = {}
  f1_scores = {}
  dataset= jload('combined-newsqa-data-v1.json')
  #data is []
  #print('how many storys:', len(dataset['data']))

  for i, article in enumerate(dataset['data']):
    storyId = article["storyId"]
    storyId  =  storyId [len("./cnn/stories/"):]   
    keeplen= len(storyId) - len(".story")
    storyId = storyId[:keeplen]
    golds = jload("answer/" + storyId + ".txt")
    preds = jload("output/" + alg + '/' + storyId + ".txt")
    for key in golds:
      gold =normalize_answer(golds[key])
      pred = normalize_answer(preds[key])
      exact_scores[key] = compute_exact(gold, pred)
      f1_scores[key] = compute_f1(gold, pred)
    #if i == 1: break
  return exact_scores, f1_scores

  

def main():
  content = ''
  for alg in ['doctalk', 'bert']:
    exact_raw, f1_raw = get_raw_scores(alg)
    out_eval = make_eval_dict(exact_raw, f1_raw)
    content += alg +':\n'
    content += 'F1 = ' +  str(out_eval['f1']) + ', exact_match = ' + str(out_eval['exact']) + '\n' 
  
  totalQ = out_eval['total']
  totalSentsList = jload( 'output/Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload('output/Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( 'output/nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/len(totalWordsList), 5)
  doctalkSummDurList = jload( 'output/DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/len(totalWordsList), 5)  
  qaDur_doctalk_self_list = jload( 'output/QA_doctalk_self_duration.json')
  avgDoctalkQaSelf = round(sum(qaDur_doctalk_self_list)/totalQ, 5)
  qaDur_doctalk_bert_list = jload( 'output/QA_doctalk_bert_duration.json')
  avgDoctalkQaBert = round(sum(qaDur_doctalk_bert_list)/totalQ, 5) 
  qaDur_BERT_bert_list = jload( 'output/QA_BERT_bert_duration.json')
  avgBertQaBert = round(sum(qaDur_BERT_bert_list)/totalQ, 5)


  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total articles: ' + str(len(totalWordsList)) + '\n'
  stats += 'average nlpParse duration per article (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per article (seconds): ' + str(avgDoctalkSummDur) + '\n' 

  stats += 'Total questions: ' + str(totalQ) + '\n'
  stats += 'average doctalk self duration per question (seconds): ' + str(avgDoctalkQaSelf) + '\n' 
  stats += 'average doctalk bert duration per question (seconds): ' + str(avgDoctalkQaBert) + '\n'
  stats += 'average BERT bert duration per question (seconds): ' + str(avgBertQaBert) + '\n' 

  print(stats )
  print("score:\n", content)

  toFile = "output/score_NewsQA.txt"
  print('save score to file:', toFile)
  
  with open(toFile,'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")

if __name__ == '__main__':
  main()

