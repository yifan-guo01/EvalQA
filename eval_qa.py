import sys
import os
import json
import time
import argparse
import glob
from doctalk.talk import Talker, nice_keys, exists_file, jload, jsave
from doctalk.params import talk_params, ropen, wopen
from doctalk.think import reason_with
from doctalk.refiner import ask_bert

qaAnswers_doctalk = dict()
qaAnswers_bert =  dict()
totalSentsList = []
totalWordsList = []
nlpParseDurList = []
doctalkSummDurList = []  
qaDur_doctalk_self_list = []
qaDur_doctalk_bert_list = []
qaDur_BERT_bert_list = []


#createSQuADQuestionIDMap('1.1') or createSQuADQuestionIDMap('2.0')
def createSQuADQuestionIDMap(version):
  datadir = "dataset/SQuAD/"  + version + "/"
  if version == "1.1":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  qidMap = dict()
  for article in dataset['data']:
      for i, paragraph in enumerate(article['paragraphs']):    
         questions = paragraph['qas']
         for question in questions:
             qid = question['id']
             q=question['question']
             qidMap[qid] = article['title']  + "_" + str(i) + "_" + q
  output = json.dumps(qidMap)
  fname = datadir + "qidMap.json"
  with wopen(fname) as f:
    f.write(output + "\n")
  f.close()

#saveSQuAD_QuestionContent('1.1') or   saveSQuAD_QuestionContent('2.0')
def saveSQuAD_QuestionContent(version):
  datadir = "dataset/SQuAD/" + version + "/"
  outputDir = datadir + "/paragraph/"
  os.makedirs(outputDir,exist_ok=True)
  os.makedirs(outputDir + 'dev',exist_ok=True)
  os.makedirs(outputDir + 'output',exist_ok=True)  
  if version == "1.1":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  for article in dataset['data']: 
      for i, paragraph in enumerate(article['paragraphs']):
         fname = outputDir + "dev/" + article['title']  + "_" + str(i) + ".txt"
         context = paragraph['context']
         with wopen(fname) as fcontext :
            fcontext.write(context + "\n")
         fcontext.close()          
         questions = paragraph['qas']
         fqname = outputDir + "dev/" + article['title']  + "_" + str(i) + "_quest.txt" 
         with wopen(fqname) as fquest:
           for question in questions:
             q=question['question']
             fquest.write(q + "\n")
           fquest.close()

 
#answerSQuADFromFile('1.1') or  answerSQuADFromFile('2.0')
def answerSQuADFromFile(version):
  datadir = "dataset/SQuAD/" + version + "/"
  if version == "1.1":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  outputDir = datadir + "/paragraph/output/"
  loadResult(outputDir)
  

  for count, article in enumerate(dataset['data']):
    for i, paragraph in enumerate(article['paragraphs']):
      #if i<3: continue
      fname = datadir +"paragraph/dev/" + article['title']  + "_" + str(i)
      doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname)
      bertAnswers, bertDur = reason_with_bert(fname)
      '''
      print('\n\ndoctalkAnswers:', doctalkAnswers)
      print('bertAnswers:', bertAnswers)
      print('Total sentences:', totalSents)
      print('Total words:', totalWords)
      print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
      print("doctalk summKeys duration(seconds): ", round(doctalkSummDur, 5))
      print("doctalk Q&A duration(seconds): ", doctalkQaDur)
      print("bert duration(seconds): ", bertDur)
      '''
      totalSentsList.append(totalSents)
      totalWordsList.append(totalWords)
      nlpParseDurList.append(nlpParseDur)
      doctalkSummDurList.append(doctalkSummDur)
      qaDur_doctalk_self_list.append(doctalkQaDur['self'])
      qaDur_doctalk_bert_list.append(doctalkQaDur['bert'])
      qaDur_BERT_bert_list.append(bertDur)

      qids = []
      for qa in paragraph['qas']:
        qid = qa['id']
        qids.append(qid)
      for j, qid in enumerate(qids):
        qaAnswers_doctalk[qid] = doctalkAnswers[j]
        qaAnswers_bert[qid] = bertAnswers[j]
        #if j == 1: break
      '''
      print('qaAnswers_doctalk:', qaAnswers_doctalk)
      print('qaAnswers_bert:', qaAnswers_bert)      
      print('totalSentsList:', totalSentsList)
      print('totalWordsList:', totalWordsList)
      print("nlpParseDurList: ", nlpParseDurList)
      print("doctalkSummDurList: ", doctalkSummDurList)
      print('qaDur_doctalk_self_list:', qaDur_doctalk_self_list)
      print('qaDur_doctalk_bert_list:', qaDur_doctalk_bert_list)
      print('qaDur_BERT_bert_list:', qaDur_BERT_bert_list)
      '''

      outputAnswers = json.dumps(qaAnswers_doctalk)
      with wopen(outputDir + 'predictions_doctalk.json' ) as fpred:
        fpred.write(outputAnswers + "\n")
      outputAnswers = json.dumps(qaAnswers_bert)
      with wopen(outputDir + 'predictions_bert.json' ) as fpred:
        fpred.write(outputAnswers + "\n")
      saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list)
      #if i == 1: break
    #if count ==0: break


#################################################################################
# above work for SQuAD paragraph, it is same test in SQuAD1.1 dev
# https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/
# evaluateSQuAD(type) can be used for wholeArticle
# below is for whole Article
#####################################################################################
#saveSQuAD_QuestionContent_wholeArticle('1.1') or saveSQuAD_QuestionContent_wholeArticle('2.0')
def saveSQuAD_QuestionContent_wholeArticle(version):
  datadir = "dataset/SQuAD/" + version + "/"
  outputdir = datadir + "/article/"
  os.makedirs(outputdir,exist_ok=True)
  os.makedirs(outputdir + 'dev',exist_ok=True)
  os.makedirs(outputdir + 'output',exist_ok=True)
  if version == "1.1":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  for article in dataset['data']:
      context = ''
      questions = '' 
      totalParagragh = len(article['paragraphs'])   
      for i, paragraph in enumerate(article['paragraphs']):
         context += paragraph['context'] + '\n'
         qas = paragraph['qas']
         for q in qas:
             questions += q['question'] + '\n'
         if i == (totalParagragh -1):
            fname = outputdir + "dev/" + article['title'] + ".txt"
            with wopen(fname) as fcontext :
              fcontext.write(context + "\n")
            fname = outputdir + "dev/" + article['title'] + "_quest.txt" 
            with wopen(fname) as f :
              f.write(questions + "\n")


# for 1.1
def answerSQuADFromFile_wholeArticle():
  datadir = "dataset/SQuAD/1.1/"
  outputDir = datadir + "article/output/"
  os.makedirs(outputDir,exist_ok=True)
  dataset= jload( datadir + "dev-v1.1.json")

  loadResult(outputDir)
  #data is []
  for count, article in enumerate(dataset['data']):
    #if count < 2: continue
    qids = []
    fname = datadir + "article/" +"dev/" + article['title'] 
    doctalkAnswers,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname)
    bertAnswers, bertDur = reason_with_bert(fname)
    ''' 
    print('doctalkAnswers:', doctalkAnswers)
    print('bertAnswers:', bertAnswers)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    print("bert duration(seconds): ", bertDur)
    '''  
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    qaDur_doctalk_self_list.append(doctalkQaDur['self'])
    qaDur_doctalk_bert_list.append(doctalkQaDur['bert'])
    qaDur_BERT_bert_list.append(bertDur)

    for i, paragraph in enumerate(article['paragraphs']):      
      for qa in paragraph['qas']:
        qid = qa['id']
        qids.append(qid)
      
    for j, qid in enumerate(qids):
      qaAnswers_doctalk[qid] = doctalkAnswers[j]
      qaAnswers_bert[qid] = bertAnswers[j]
      #if j ==1: break
    '''
    print('\n\nDone, save to files')
    print('qaAnswers_doctalk:', qaAnswers_doctalk)
    print('qaAnswers_bert:', qaAnswers_bert)   
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('qaDur_doctalk_self_list:', qaDur_doctalk_self_list)
    print('qaDur_doctalk_bert_list:', qaDur_doctalk_bert_list)
    print('qaDur_BERT_bert_list:', qaDur_BERT_bert_list)
    '''
    
    outputAnswers = json.dumps(qaAnswers_doctalk)
    with wopen(outputDir + 'predictions_doctalk.json' ) as fpred:
      fpred.write(outputAnswers + "\n")
    outputAnswers = json.dumps(qaAnswers_bert)
    with wopen(outputDir + 'predictions_bert.json' ) as fpred:
      fpred.write(outputAnswers + "\n")

    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list)
    #if count == 1: break
    

##########################################################################################
#for NewsQA
#########################################################################################

def saveNewQA_QuestionContent():
  datadir = "dataset/NewsQA/"
  os.makedirs(datadir + 'dev',exist_ok=True)
  os.makedirs(datadir + 'answer',exist_ok=True)

  dataset= jload( datadir + 'combined-newsqa-data-v1.json')
  #data is []
  print('how many storys:', len(dataset['data']))

  for i, article in enumerate(dataset['data']):
    storyId = article["storyId"]
    storyId  =  storyId [len("./cnn/stories/"):]   
    keeplen= len(storyId) - len(".story")
    storyId = storyId[:keeplen]    
    
    fname = datadir + "dev/" + storyId + ".txt"
    conext = article['text']
    with wopen(fname) as fcontext :
      fcontext.write(conext + "\n")
          
    questions = article['questions']
    qstring = ""
    astring = ""
    answerMap =dict()  
    
    for j, question in enumerate(questions):
      q=question['q']
      qstring += q + "\n"
      a=question['consensus']
      #print('a:', a)
      if 'badQuestion' in a.keys():
        #print("*****find bad question")
        answer = ""
      elif 'noAnswer' in a.keys():
        #print("******noAnswer")
        answer = ""
      else:
        start = a['s']
        end = a['e']
        #print('start:end, ', start, end )
        answer = conext[a['s']:a['e']]
      #print('answer:', answer)
      answerMap[storyId + "_" + str(j)] = answer
      

    fqname = datadir + "dev/" + storyId + "_quest.txt" 
    with wopen(fqname) as fquest:
      fquest.write(qstring + "\n")
    
    faname = datadir + "answer/" + storyId + ".txt"
    outputAnswer = json.dumps(answerMap)
    with wopen(faname) as fanswer:
      fanswer.write( outputAnswer + "\n" )


def answerNewsQA():
  datadir = "dataset/NewsQA/"
  outputDir = 'dataset/NewsQA/output/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'doctalk',exist_ok=True)
  os.makedirs(outputDir + 'bert',exist_ok=True)
  dataset= jload( datadir + 'combined-newsqa-data-v1.json')

  loadResult(outputDir)

  for i, article in enumerate(dataset['data']):
    #if i < 1: continue
    storyId = article["storyId"]
    storyId  =  storyId [len("./cnn/stories/"):]
    keeplen= len(storyId) - len(".story")
    storyId = storyId[:keeplen]
    fname = datadir + "dev/" + storyId  
    doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname)
    bertAnswers, bertDur = reason_with_bert(fname)
    '''
    print('doctalkAnswers:', doctalkAnswers)
    print('bertAnswers:', bertAnswers)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    print("bert duration(seconds): ", bertDur)
    ''' 
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    qaDur_doctalk_self_list.append(doctalkQaDur['self'])
    qaDur_doctalk_bert_list.append(doctalkQaDur['bert'])
    qaDur_BERT_bert_list.append(bertDur)

    
    questions = article['questions']
    qaAnswers_doctalk =dict()  
    qaAnswers_bert = dict()

    for j, question in enumerate(questions):
      qaAnswers_doctalk[storyId + "_" + str(j)] = doctalkAnswers[j]
      qaAnswers_bert[storyId + "_" + str(j)] = bertAnswers[j]
    
    fname = outputDir + "doctalk/" + storyId + ".txt"
    outputAnswers = json.dumps(qaAnswers_doctalk)
    with wopen(fname) as fpred:
      fpred.write(outputAnswers + "\n")

    fname = outputDir + "bert/" + storyId + ".txt"
    outputAnswers = json.dumps(qaAnswers_bert)
    with wopen(fname) as fpred:
      fpred.write(outputAnswers + "\n")
    
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list)
    #if i==1:break

###########################################################################################################################
### below if for narrativeqa
##############################################################################################################################
def saveNarrativeqa_QuestionContent():
  baseDir = 'dataset/Narrativeqa/'
  os.makedirs(baseDir + 'dev/', exist_ok=True)
  os.makedirs(baseDir + 'output/', exist_ok=True)
  gitDir = baseDir + '/narrativeqa_github/'
  import csv
  with open(gitDir + '/third_party/wikipedia/summaries.csv', newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    for row in dataset:
      if row['set'] == 'test':
        fname = baseDir + "dev/" + row['document_id'] + ".txt"
        with wopen(fname) as fcontext :
          fcontext.write(row['summary'] + '\n')

  from collections import defaultdict
  dqs = defaultdict(list)
  with open(gitDir + 'qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] == 'test':
        dqs[row['document_id']].append(row['question'])

  for id in dqs:
    questions = "\n".join(dqs[id])
    fname = baseDir + "dev/" + id + "_quest.txt"
    with wopen(fname) as f :
      f.write(questions + '\n')
 

def answerNarrativeqa():
  baseDir = "dataset/Narrativeqa/"
  outputDir = baseDir + 'output/'
  gitDir = baseDir + 'narrativeqa_github/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'doctalk',exist_ok=True)
  os.makedirs(outputDir + 'bert',exist_ok=True)

  loadResult(outputDir)
  import csv
  from collections import defaultdict
  dataIds = []
  with open(gitDir + '/third_party/wikipedia/summaries.csv', newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    for row in dataset:
      if row['set'] != 'test': continue
      dataIds.append(row['document_id'])
  dqs = defaultdict(list)
  with open(gitDir + 'qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] != 'test': continue
      dqs[row['document_id']].append(row['question']) 
  i = 0
  for document_id in dataIds:
    fname = baseDir + "dev/" + document_id
    if os.path.exists(fname + '.txt') == False:
      continue
    doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname)
    bertAnswers, bertDur = reason_with_bert(fname)
    '''
    print('doctalkAnswers:', doctalkAnswers)
    print('bertAnswers:', bertAnswers)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    print("bert duration(seconds): ", bertDur)
    '''

    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    qaDur_doctalk_self_list.append(doctalkQaDur['self'])
    qaDur_doctalk_bert_list.append(doctalkQaDur['bert'])  
    qaDur_BERT_bert_list.append(bertDur)

    qaAnswers_doctalk = {}
    qaAnswers_bert = {}
    for j, question in enumerate(dqs[document_id]):
      qaAnswers_doctalk[question] = doctalkAnswers[j]
      qaAnswers_bert[question] = bertAnswers[j]
    
    '''
    print('\n\nDone, save to files')
    print('qaAnswers_doctalk:', qaAnswers_doctalk)
    print('qaAnswers_bert:', qaAnswers_bert)
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('qaDur_doctalk_self_list:', qaDur_doctalk_self_list)
    print('qaDur_doctalk_bert_list:', qaDur_doctalk_bert_list)
    print('qaDur_BERT_bert_list:', qaDur_BERT_bert_list) 
    '''
    fname = outputDir + "doctalk/" + document_id + ".txt"
    outputAnswers = json.dumps(qaAnswers_doctalk)
    with wopen(fname) as fpred:
      fpred.write(outputAnswers + "\n")

    fname = outputDir + "bert/" + document_id + ".txt"
    outputAnswers = json.dumps(qaAnswers_bert)
    with wopen(fname) as fpred:
      fpred.write(outputAnswers + "\n")
    
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list)
    i = i+1
    #if i==2:break


###########################################################################################################################
### below if for HotpotQA
##############################################################################################################################
def saveHotpotQA_QuestionContent():
  os.makedirs('dataset/HotpotQA/dev/', exist_ok=True)
  os.makedirs('dataset/HotpotQA/answer/', exist_ok=True)
  dataset= jload('dataset/HotpotQA/hotpot_dev_distractor_v1.json')  
  #data is []
  print('dataset length:', len(dataset))
  for i, article in enumerate(dataset):    
    quest_id = article["_id"]
    #print('quest_id:', quest_id)
    question = article["question"]
    #print('question ', i, ':', question)
    fqname = "dataset/HotpotQA/dev/" + quest_id + "_quest.txt" 
    with wopen(fqname) as fquest:
        fquest.write(question + "\n")
    
    answer = article["answer"]
    fqname = "dataset/HotpotQA/answer/" + quest_id + ".txt" 
    with wopen(fqname) as fanswer:
      fanswer.write(answer + "\n")

    text = ''
    paralist = article["context"]
    #print('paralist type:', type(paralist))
    for para in paralist:
      #title = para[0]
      #print('\n\ntitle:', title)
      text += '\n'
      sentences = para[1]
      #print('sentences:', sentences)
      for sent in sentences:
        text += sent
    
    fname = "dataset/HotpotQA/dev/" + quest_id + ".txt"
    with wopen(fname) as fcontext :
      fcontext.write(text + '\n')
    #if i > 10 : break  


def answerHotpotQA():
  outputDir = 'dataset/HotpotQA/output/'
  os.makedirs(outputDir, exist_ok=True)
  dataset= jload('dataset/HotpotQA/hotpot_dev_distractor_v1.json')
  #data is []
  loadResult(outputDir)

  for i, article in enumerate(dataset): 
    #if i<12: continue   
    quest_id = article["_id"]
    fname = "dataset/HotpotQA/dev/" + quest_id
    doctalkAnswers,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname)
    bertAnswers, bertDur = reason_with_bert(fname)
    '''
    print('doctalkAnswers:', doctalkAnswers)
    print('bertAnswers:', bertAnswers)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    print("Bert Q&A duration(seconds): ", bertDur)
    '''
    qaAnswers_doctalk[quest_id] = doctalkAnswers[0]
    qaAnswers_bert[quest_id] = bertAnswers[0]
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    qaDur_doctalk_self_list.append(doctalkQaDur['self'])
    qaDur_doctalk_bert_list.append(doctalkQaDur['bert'])
    qaDur_BERT_bert_list.append(bertDur)

    '''
    print('\n\nDone, save to files')
    print('qaAnswers_doctalk:', qaAnswers_doctalk)
    print('qaAnswers_bert:', qaAnswers_bert)   
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('qaDur_doctalk_self_list:', qaDur_doctalk_self_list)
    print('qaDur_doctalk_bert_list:', qaDur_doctalk_bert_list)
    print('qaDur_BERT_bert_list:', qaDur_BERT_bert_list)   
    '''
    outputAnswers = json.dumps(qaAnswers_doctalk)
    with wopen(outputDir + 'predictions_doctalk.json' ) as fpred:
      fpred.write(outputAnswers + "\n")
    outputAnswers = json.dumps(qaAnswers_bert)
    with wopen(outputDir + 'predictions_bert.json' ) as fpred:
      fpred.write(outputAnswers + "\n")

    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list )
    #if i==2 : break


###########################################################################################################################
### below if for biased_textrank
# saveTextrank_QuestionContent('test')  or saveTextrank_QuestionContent('val')
# saveTextrank_QuestionContent('test2')  or saveTextrank_QuestionContent('val2')
##############################################################################################################################
def saveTextrank_QuestionContent(type):
  testpath = 'dataset/textrank/' + type + '/'
  os.makedirs(testpath, exist_ok=True)
  os.makedirs(testpath + 'dev/', exist_ok=True)
  os.makedirs(testpath + 'answer/', exist_ok=True)
  os.makedirs(testpath + 'answer_biased/', exist_ok=True)
  os.makedirs(testpath + 'answer_gpt2/', exist_ok=True)
  dataset= jload('dataset/textrank/biased_textrank_git/data/liar/clean_' + type + '.json')  
  #data is []
  print('dataset length:', len(dataset))
  for article in dataset:    
    document_id = article["id"]
    question = article["claim"].replace('&nbsp;', '').strip()
    text = article["statements"].replace('&nbsp;', '').strip()
    answer = article["new_justification"].replace('&nbsp;', '').strip()
    answer_biased = article["generated_justification_biased"].replace('&nbsp;', '').strip()
    answer_gpt2 = article["generated_justification_gpt2"].replace('&nbsp;', '').strip()       
    if len(question) == 0  or len(text) == 0 or len(answer) == 0:
      print(document_id, ', question length=', len(question), ', text length=', len(text), ', answer length= ', len(answer) )
      continue
    if len(answer_biased) == 0 or len(answer_gpt2) == 0:
       print(document_id, ', answer_biased length=', len(answer_biased), ', answer_gpt2 length=', len(answer_gpt2) ) 
       continue 
    fqname = testpath + 'dev/' + str(document_id) + "_quest.txt" 
    with wopen(fqname) as f:
        f.write(question + "\n") 
    fname = testpath + 'dev/' + str(document_id) + ".txt"
    with wopen(fname) as f :
      f.write(text + '\n')     
    fqname = testpath + 'answer/' + str(document_id) + ".txt" 
    with wopen(fqname) as f:
      f.write(answer + "\n")
    fqname = testpath + 'answer_biased/' + str(document_id) + ".txt" 
    with wopen(fqname) as f:
        f.write(answer_biased + "\n") 
    fqname = testpath + 'answer_gpt2/' + str(document_id) + ".txt" 
    with wopen(fqname) as f:
        f.write(answer_gpt2 + "\n") 

#answerTextrank('test') or answerTextrank('val')
#answerTextrank('test2') or answerTextrank('val2')
def answerTextrank(type):
  outputDir = 'dataset/textrank/' + type + '/output/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'predictions',exist_ok=True)
  loadResult(outputDir)
  
  doc_files = glob.glob('dataset/textrank/' + type + "/answer/*.txt")
  for doc_file in doc_files:
    fname=doc_file.split('/')[-1][0: -4]
    fname = 'dataset/textrank/' + type + '/dev/' + fname
    print(fname)
    if os.path.exists(fname + '.txt') == False:
      continue
    doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk(fname, askBert=0)
    '''  
    print('doctalkAnswers:', doctalkAnswers)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''  
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    qaDur_doctalk_self_list.append(doctalkQaDur['self'])

    fname = outputDir + "predictions/" + str(id) + ".txt"
    with wopen(fname) as fpred:
      fpred.write(doctalkAnswers[0] + "\n")
    '''
    print('\n\nDone, save to files')
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('qaDur_doctalk_self_list:', qaDur_doctalk_self_list)
    '''
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        qaDur_doctalk_self_list )
    #break


#lead4_Textrank('test') or lead4_Textrank('val')
#lead4_Textrank('test2') or lead4_Textrank('val2')
def lead4_Textrank(type):  
  outputDir = 'dataset/textrank/' + type + '/lead4/'
  os.makedirs(outputDir, exist_ok=True)  
  doc_files = glob.glob('dataset/textrank/' + type + "/answer/*.txt")
  for doc_file in doc_files:
    document_ID=doc_file.split('/')[-1][0: -4]
    fname = 'dataset/textrank/' + type + '/dev/' + document_ID + '.json'
    #print(fname)
    if os.path.exists(fname) == False:
      continue
    db = jload(fname)
    totsents = len(db[0])
    #print('total sentences:', totsents)
    SENT,LEMMA,TAG,NER,BEFORE,DEP,IE=0,1,2,3,4,5,6
    txt = ""
    for i in range(0, 4):
      sent = db[0][i][SENT]
      #print('sent:', sent)
      before = db[0][i][BEFORE]
      ws = ""
      for j in range(len(sent)):
        ws += before[j] + sent[j]
      #print('make to sentence:\n', ws)
      txt += ' ' + ws
      if i== totsents-1:
        print(fname, ' has total', totsents, 'sentences, beak')
        break
    #print('Get the first 4 sentences:\n', txt)
    fname = outputDir + document_ID + '.txt'
    #print('save to ', fname)
    with wopen(fname) as f:
      f.write(txt + "\n")



def reason_with_doctalk(fname, askBert=0.1) :  
  params = talk_params()
  params.with_answerer=False #True: get think answer, False: get talk answer
  params.top_answers = 4
  params.max_answers = 4
  params.with_bert_qa = askBert
  params.answers_by_rank = True
  params.stanza_parsing = False
  doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSumDur, doctalkQaDur= reason_with(fname, params=params)
  return doctalkAnswers, totalSents, totalWords, nlpParseDur, doctalkSumDur, doctalkQaDur


def reason_with_bert(fname):
    startTime = time.time()
    with open(fname + '.txt','r',encoding='utf8') as f:
        content=f.read()
    with open(fname + '_quest.txt','r',encoding='utf8') as f:
        qs = list(l.strip() for l in f)
    answers = []
    for q in qs :
        if not q :break
        print("\n===========>QUESTION: ", q)
        r = ask_bert(content, q, 0.0001)
        if not r :
            r = ''
        removes = [",",".", "?", "!", ",\"", ".\"","?\"", "!\"", "'s"]
        for remove in removes:
            if r.endswith(remove):
                keepLen = len(r) - len(remove)
                r = r[:keepLen]  
        print("\n===========>BERT SHORT ANSWER:",r,'<===========\n')
        answers.append(r)
    endTime = time.time()
    duration = endTime - startTime
    return answers, duration
  

def saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
                  nlpParseDurList, doctalkSummDurList,
                  qss, qbs=[], bertDurs=[] ):
  outputTotalSents = json.dumps(totalSentsList)
  with wopen(outputDir + 'Total_Sents.json' ) as f:
    f.write(outputTotalSents + "\n")

  outputTotalWords = json.dumps(totalWordsList)
  with wopen(outputDir + 'Total_Words.json' ) as f:
    f.write(outputTotalWords + "\n")

  outputNlpParseDur = json.dumps(nlpParseDurList)
  with wopen(outputDir + 'nlpParse_duration.json' ) as f:
    f.write(outputNlpParseDur + "\n")

  outputDoctalkSummDur = json.dumps(doctalkSummDurList)
  with wopen(outputDir + 'DoctalkSumm_duration.json' ) as f:
    f.write(outputDoctalkSummDur + "\n")

  with wopen(outputDir + 'QA_doctalk_self_duration.json' ) as f:
    f.write(json.dumps(qss) + "\n")
  if len(qbs)>1:
    with wopen(outputDir + 'QA_doctalk_bert_duration.json' ) as f:
      f.write(json.dumps(qbs) + "\n")
  if len(bertDurs)>1:
    with wopen(outputDir + 'QA_BERT_bert_duration.json' ) as f:
      f.write(json.dumps(bertDurs) + "\n")

def loadResult(outputdir):
  global qaAnswers_doctalk, qaAnswers_bert
  global totalSentsList, totalWordsList
  global nlpParseDurList, doctalkSummDurList
  global qaDur_doctalk_self_list, qaDur_doctalk_bert_list, qaDur_BERT_bert_list


  if os.path.exists(outputdir +  'predictions_doctalk.json'):
    qaAnswers_doctalk = jload( outputdir +  'predictions_doctalk.json')
  if os.path.exists(outputdir +  'predictions_bert.json'):
    qaAnswers_bert = jload( outputdir +  'predictions_bert.json')
  if os.path.exists(outputdir +  'Total_Sents.json'):
    totalSentsList = jload( outputdir +  'Total_Sents.json')
  if os.path.exists(outputdir +  'Total_Words.json'):
    totalWordsList = jload( outputdir +  'Total_Words.json')
  if os.path.exists(outputdir +  'nlpParse_duration.json'):
    nlpParseDurList = jload( outputdir +  'nlpParse_duration.json')
  if os.path.exists(outputdir +  'DoctalkSumm_duration.json'):
    doctalkSummDurList = jload( outputdir +  'DoctalkSumm_duration.json') 
  if os.path.exists(outputdir +  'QA_doctalk_self_duration.json'):
    qaDur_doctalk_self_list =  jload( outputdir +  'QA_doctalk_self_duration.json') 
  if os.path.exists(outputdir +  'QA_doctalk_bert_duration.json'):
    qaDur_doctalk_bert_list =  jload( outputdir +  'QA_doctalk_bert_duration.json') 
  if os.path.exists(outputdir +  'QA_BERT_bert_duration.json'):
    qaDur_BERT_bert_list =  jload( outputdir +  'QA_BERT_bert_duration.json') 

  print('loadSQuADResult:')
  print('qaAnswers_doctalk:', qaAnswers_doctalk)
  print('qaAnswers_bert:', qaAnswers_bert)
  print('totalSentsList:', totalSentsList)
  print('totalWordsList:', totalWordsList)
  print("nlpParseDurList: ", nlpParseDurList)
  print("doctalkSummDurList: ", doctalkSummDurList)
  print("qaDur_doctalk_self_list: ", qaDur_doctalk_self_list)
  print("qaDur_doctalk_bert_list: ", qaDur_doctalk_bert_list)
  print("qaDur_BERT_bert_list: ", qaDur_BERT_bert_list)

if __name__ == '__main__' :
  pass



