import collections
import json
import os
import re
import string
import sys
import csv
import rouge_stats as rs
import glob

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

def score2txt(files, p, r, f) :
  zipped = zip(files, p, r, f)
  txt = ''
  for item in zipped :
    file,pre,recall,fm=item
    txt += file + ', '
    txt += str(pre) + ', '
    txt += str(recall) + ', '
    txt += str(fm) + '\n'
  return txt

def avg(xs) :
  s=sum(xs)
  l=len(xs)
  if 0==l : return None
  return s/l  


def eval_with_rouge(type) :
  files=[]
  f1=[]
  p1=[]
  r1=[] 
  f2=[]
  p2=[]
  r2=[] 
  fl=[]
  pl=[]
  rl=[] 
  ref_dir = type + "/answer/"
  pred_dir = type + "/output/predictions/"
  print('ref_dir:', ref_dir)
  print('pred_dir:', pred_dir)
  doc_files = sorted(glob.glob(type + "/answer/*.txt"))
  print(len(doc_files))
  j = 0
  for doc_file in doc_files : 
    fname=doc_file.split('/')[-1]
    ref_name=ref_dir+fname
    pred_name=pred_dir+fname
    '''
    print('fname:',fname)
    print('ref_name:', ref_name)
    print('rouge', i, ', pred_name:', pred_name)
    '''

    with open(ref_name,'r',encoding='utf8') as fgold:
      gold=fgold.read()   
    with open(pred_name,'r',encoding='utf8') as fsliver:
      silver=fsliver.read() 
    if not gold:
      print('gold file missing:', ref_name)
      continue
    if not silver:
      print('silver file missing:', pred_name)
      continue

    files.append(fname)
    '''
    print('gold:\n', gold )
    print('silver:\n', silver )
    '''
    res = rs.rstat('rouge', silver,gold)
    #print('res:\n', *res)
    #print('res[0]:\n',res[0])
    
    p1.append(res[0]['rouge-1']['p'])
    r1.append(res[0]['rouge-1']['r'])
    f1.append(res[0]['rouge-1']['f'])
    p2.append(res[0]['rouge-2']['p'])
    r2.append(res[0]['rouge-2']['r'])
    f2.append(res[0]['rouge-2']['f'])
    pl.append(res[0]['rouge-l']['p'])
    rl.append(res[0]['rouge-l']['r'])
    fl.append(res[0]['rouge-l']['f'])
    '''
    print('p1:\n', p1)
    print('r1:\n', r1)
    print('f1:\n', f1)
    print('p2:\n', p2)
    print('r2:\n', r2)
    print('f2:\n', f2)
    print('pl:\n', pl)
    print('rl:\n', rl)
    print('fl:\n', fl)
    '''
    j = j+1
    #if j==2: break

  p = []
  p.append(p1)
  p.append(p2)
  p.append(pl)
  r = []
  r.append(r1)
  r.append(r2)
  r.append(rl)
  f = []
  f.append(f1)
  f.append(f2)
  f.append(fl)
  '''
  print('p:/n', p)
  print('r:/n', r)
  print('f:/n', f)
  '''
  rouge_names=('ROUGE_1','ROUGE_2','ROUGE_l')
  for i, rouge_name in enumerate(rouge_names) :
    #print (rouge_name,':', ', Precision=', avg(p[i]),  ', Recall=', avg(r[i]), ', F-Measure=', avg(f[i]))

    #save ABS ROUGE scores into file
    content = 'fileName, Precision, Recall, F-Measure' + '\n'
    content += score2txt(files, p[i], r[i], f[i])
    #print('content:\n',content )
  
    toFile = "Abs" + rouge_name + ".csv"
    with open(pred_dir + toFile,'w',encoding='utf8') as frouge:
      frouge.write(content + "\n")
  
  return ((avg(p1),avg(r1),avg(f1)), (avg(p2),avg(r2),avg(f2)), (avg(pl),avg(rl),avg(fl)))


def main():
  if len(sys.argv) != 2 or  sys.argv[1] not in ['test', 'test2', 'val', 'val2' ]:
    print('Run one of the commands as below:')
    print(' python evaluate.py test')
    print(' python evaluate.py test2')
    print(' python evaluate.py val ')
    print(' python evaluate.py val2 ')
    sys.exit(0)
  type = sys.argv[1]
  content = ''
  r1, r2, rl = eval_with_rouge(type )
  '''
  print('r1:', r1)
  print('r2:', r2)
  print('rl:', rl)
  '''
  content += 'ROUGE_1 F-Measure= '+ str(round(r1[2], 5)) 
  content += ', ROUGE_2 F-Measure= '+ str(round(r2[2],5))
  content +=  ', ROUGE_L F-Measure= ' + str(round(rl[2], 5)) + '\n'
  

  doc_files = glob.glob(type + "/answer/*.txt")
  totalQ = len(doc_files)
  print('totalQ=', totalQ)

  outDir = type + '/output/'
  totalSentsList = jload( outDir + 'Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload( outDir + 'Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( outDir + 'nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/len(totalWordsList), 5)
  doctalkSummDurList = jload( outDir + 'DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/len(totalWordsList), 5)  
  qaDur_doctalk_self_list = jload( outDir + 'QA_doctalk_self_duration.json')
  avgDoctalkQaSelf = round(sum(qaDur_doctalk_self_list)/totalQ, 5)


  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total articles: ' + str(len(totalWordsList)) + '\n'
  stats += 'average nlpParse duration per article (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per article (seconds): ' + str(avgDoctalkSummDur) + '\n' 

  stats += 'Total questions: ' + str(totalQ) + '\n'
  stats += 'average doctalk self duration per question (seconds): ' + str(avgDoctalkQaSelf) + '\n' 

  print(stats )
  print("score:\n", content)

  toFile = outDir + "score_textrank.txt"
  print('save score to file:', toFile)
  
  with open(toFile,'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")

if __name__ == '__main__':
  main()

