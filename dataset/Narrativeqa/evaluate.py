import collections
import json
import os
import re
import string
import sys
import csv
import rouge_stats as rs
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

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


def normalize_answer_2(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_punc(lower(s)))


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


def get_rougeL_score( pred, gold):
  k=0
  for i, res in enumerate(rs.rstat(pred,gold)) :
    if i == 2:
      d=res[0]      
      fx=d['f'][0] 
      return fx     


def get_scores(alg):
  exact_scores = {}
  f1_scores = {}
  rougel_scores = {}
  bleu1_scores = {}
  bleu4_scores = {}
  meteor_scores = {}
  ref_answers_1 = {}
  ref_answers_2 = {}
  pred_answers = {}
  outputDir ="output/"
  i = 0
  with open('narrativeqa_github/qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] != 'test': continue
      document_id = row['document_id']
      question = row['question']
      gold_answer1 =  normalize_answer(row['answer1'])
      gold_answer2 =  normalize_answer(row['answer2'])
      if os.path.exists(outputDir  + alg + '/' + document_id + ".txt") == False:
        continue
      preds = jload(outputDir + alg + '/' + document_id + ".txt")    
      pred_answer = normalize_answer(preds[question])
      questionId = document_id + '_' + question
      ref_answers_1[questionId] = row['answer1']
      ref_answers_2[questionId] = row['answer2']
      pred_answers[questionId] = preds[question]

      #for F1
      em_1 = compute_exact(gold_answer1, pred_answer)
      f1_1 = compute_f1(gold_answer1, pred_answer)
      em_2 = compute_exact(gold_answer2, pred_answer)
      f1_2 = compute_f1(gold_answer2, pred_answer)

      if f1_1 >= f1_2:
        exact_scores[questionId] = em_1
        f1_scores[questionId] = f1_1
      else:
        exact_scores[questionId] = em_2
        f1_scores[questionId] = f1_2 
      
      #for rouge_L
      fm_1 = get_rougeL_score(pred_answer, gold_answer1)
      fm_2 = get_rougeL_score(pred_answer, gold_answer2)
      rougel_scores[questionId] = max(fm_1, fm_2)
      
      gold_nAnswer1 = normalize_answer(row['answer1'])
      gold_nAnswer2 = normalize_answer(row['answer2'])
      pred_nAnswer = normalize_answer(preds[question])
      gold1_list = gold_nAnswer1.split(' ')
      gold2_list = gold_nAnswer2.split(' ') 
      pred_list =  pred_nAnswer.split(' ')
      gold_len = min(len(gold1_list), len(gold2_list))

      bleu_1 = sentence_bleu([gold1_list, gold2_list] , pred_list, weights=(1, 0, 0, 0))
      bleu1_scores[questionId] = bleu_1
      if gold_len >= 4:
        bleu_4 = sentence_bleu([gold1_list, gold2_list] , pred_list, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4_scores[questionId] = bleu_4
      meteor = meteor_score([gold_nAnswer1, gold_nAnswer2], pred_nAnswer)
      meteor_scores[questionId] = meteor


      i += 1
      #if i == 2: break

  score =dict()
  total = len(exact_scores)
  score['total'] = total
  score['f1'] = 100.0 * sum(f1_scores.values()) / len(f1_scores)
  score['exact'] = 100.0 * sum(exact_scores.values()) / len(exact_scores)
  score['rouge_l'] = 100.0 * sum(rougel_scores.values()) / len(rougel_scores)
  score['bleu_1'] = 100.0 * sum(bleu1_scores.values()) / len(bleu1_scores)
  score['bleu_4'] = 100.0 * sum(bleu4_scores.values()) / len(bleu4_scores)
  score['meteor'] = 100.0 * sum(meteor_scores.values()) / len(meteor_scores)


  content = 'question, f1, exact, rouge_L, bleu1, bleu4, meteor, ref_answer1, ref_answer2, pred_answer' + '\n'
  for quest in f1_scores :
    content += quest.replace(',', ' ').replace('\n', '') + ', '
    content += str(round(f1_scores[quest], 5)) + ', '
    content += str(round(exact_scores[quest], 5)) + ', '
    content += str(round(rougel_scores[quest], 5)) + ','
    content += str(round(bleu1_scores[quest], 5)) + ','
    if quest in bleu4_scores.keys():   
      content += str(round(bleu4_scores[quest], 5)) + ','   
    else:
      content += ' ' + ',' 
    content += str(round(meteor_scores[quest], 5)) + ','   
    content += ref_answers_1[quest].replace(',', ' ').replace('\n', '') + ','   
    content += ref_answers_2[quest].replace(',', ' ').replace('\n', '')  + ','   
    content += pred_answers[quest].replace(',', ' ').replace('\n', '') + '\n'   
  with open(outputDir + alg + '/score_detail.csv','w',encoding='utf8') as fscore:
    fscore.write(content + "\n")

  return score





def main():
  content = ''
  for alg in ['doctalk', 'bert']:
    eval = get_scores(alg)
    content += alg + ':\n'
    content += ' F1 = ' + str(eval['f1']) + ', exact_match = ' + str(eval['exact']) + '\n'
    content += ' rouge_l=' + str(eval['rouge_l']) + '\n'  
    content += ' bleu_1=' + str(eval['bleu_1']) + '\n'  
    content += ' bleu_4=' + str(eval['bleu_4']) + '\n'
    content += ' meteor=' + str(eval['meteor']) + '\n'   

  outputDir ="output/"
  totalQ = eval['total']
  totalSentsList = jload( outputDir + 'Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload(outputDir + 'Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( outputDir + 'nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/len(totalWordsList), 5)
  doctalkSummDurList = jload( outputDir + 'DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/len(totalWordsList), 5)  
  qaDur_doctalk_self_list = jload( outputDir + 'QA_doctalk_self_duration.json')
  avgDoctalkQaSelf = round(sum(qaDur_doctalk_self_list)/totalQ, 5)
  qaDur_doctalk_bert_list = jload( outputDir + 'QA_doctalk_bert_duration.json')
  avgDoctalkQaBert = round(sum(qaDur_doctalk_bert_list)/totalQ, 5) 
  qaDur_BERT_bert_list = jload( outputDir + 'QA_BERT_bert_duration.json')
  avgBertQaBert = round(sum(qaDur_BERT_bert_list)/totalQ, 5) 


  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total articles: ' + str(len(totalWordsList)) + '\n'
  stats += 'Total questions: ' + str(totalQ) + '\n'
  stats += 'average nlpParse duration per article (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per article (seconds): ' + str(avgDoctalkSummDur) + '\n' 
  stats += 'average doctalk self duration per question (seconds): ' + str(avgDoctalkQaSelf) + '\n' 
  stats += 'average doctalk bert duration per question (seconds): ' + str(avgDoctalkQaBert) + '\n' 
  stats += 'average BERT bert duration per question (seconds): ' + str(avgBertQaBert) + '\n' 


  print(stats )
  print("score:\n", content)

  toFile = outputDir + "score_Narrativeqa.txt"
  print('save score to file:', toFile)
  
  with open(toFile,'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")

if __name__ == '__main__':
  main()

