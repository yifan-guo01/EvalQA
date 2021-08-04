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

def eval_with_rouge(type, i) :
  files=[]
  f=[]
  p=[]
  r=[] 
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
    '''
    print('gold:\n', gold )
    print('silver:\n', silver )
    '''
    k=0
    for res in rs.rstat(silver,gold) :
      if k==i:    
        d=res[0]
      
        px=d['p'][0]
        rx=d['r'][0]
        fx=d['f'][0]
        files.append(fname)
        p.append(px)
        r.append(rx)
        f.append(fx)
        #print('ROUGE ( P  R  F)', fname, p, r, f )
      elif k>i : break
      k+=1
    j = j+1
    #if j==2: break
  rouge_name=(1,2,'l')  
  #print ("ROUGE",rouge_name[i],':', ', Precision=', avg(p),  ', Recall=', avg(r), ', F-Measure=', avg(f))

  #save ABS ROUGE scores into file
  content = 'fileName, Precision, Recall, F-Measure' + '\n'
  content += score2txt(files, p, r, f)
  
  toFile = "AbsRouge_" + str(rouge_name[i]) + ".csv"
  with open(pred_dir + toFile,'w',encoding='utf8') as frouge:
    frouge.write(content + "\n")
  return (avg(p),avg(r),avg(f))


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
  r1 = eval_with_rouge(type, 0 )  # 1
  r2 = eval_with_rouge(type, 1)  # 2
  rl = eval_with_rouge(type, 2)  # l
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

