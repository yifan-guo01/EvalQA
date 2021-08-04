""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

  
def jload(infile) :
  ''' loads .json file, preprocessed from a .txt file'''
  with open(infile,'r',encoding='utf8') as f:
    res = json.load(f)
    return res

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for aindex, article in enumerate(dataset):
        for pindex, paragraph in enumerate(article['paragraphs']):
            for qindex, qa in enumerate(paragraph['qas']):
                total += 1
                if qa['id'] not in predictions:
                    message = article['title'] + ', Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                #if qindex == 1: break
            #if pindex ==0: break
        #if aindex == 0: break
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

##########################################################################
#Can be used for paragraph or whole article
#python evaluate.py paragraph
#python evaluate.py article
##########################################################################

if __name__ == '__main__':
  if len(sys.argv) != 2 or  sys.argv[1] not in ['paragraph', 'article' ]:
    print('Run one of the commands as below:')
    print(' python evaluate.py paragraph')
    print(' python evaluate.py article ')
    sys.exit(0)
  type = sys.argv[1]
  dataset= jload( "dev-v1.1.json")
  outputDir = type + '/output/'
  content = ''
  for alg in ['doctalk', 'bert']:
    predictions = jload( outputDir + 'predictions_' +  alg + '.json')
    score = evaluate(dataset['data'], predictions)
    em = round(score['exact_match'], 2)
    f1 = round(score['f1'], 2)
    content += alg + ':\n' 
    content += 'F1 = ' +  str(f1) + ', exact_match = ' + str(em) + '\n'

  totalSentsList = jload( outputDir + 'Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload( outputDir + 'Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( outputDir + 'nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/len(totalWordsList), 5)
  doctalkSummDurList = jload( outputDir + 'DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/len(totalWordsList), 5) 

  qaDur_doctalk_self_list = jload( outputDir + 'QA_doctalk_self_duration.json')
  avgDoctalkQaSelf = round(sum(qaDur_doctalk_self_list)/len(predictions), 5)
  qaDur_doctalk_bert_list = jload( outputDir + 'QA_doctalk_bert_duration.json')
  avgDoctalkQaBert = round(sum(qaDur_doctalk_bert_list)/len(predictions), 5) 
  qaDur_BERT_bert_list = jload( outputDir + 'QA_BERT_bert_duration.json')
  avgBertQaBert = round(sum(qaDur_BERT_bert_list)/len(predictions), 5)


  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total articles: ' + str(len(totalWordsList)) + '\n'
  stats += 'average nlpParse duration per article (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per article (seconds): ' + str(avgDoctalkSummDur) + '\n' 

  stats += 'Total questions: ' + str(len(predictions)) + '\n'
  stats += 'average doctalk self duration per question (seconds): ' + str(avgDoctalkQaSelf) + '\n' 
  stats += 'average doctalk bert duration per question (seconds): ' + str(avgDoctalkQaBert) + '\n' 
  stats += 'average BERT bert duration per question (seconds): ' + str(avgBertQaBert) + '\n' 


  print(stats )
  print("score:\n", content)

  toFile = outputDir + "SQuAD_1.1_score.txt"
  print('save score to file:', toFile)
  with open(toFile, 'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")
