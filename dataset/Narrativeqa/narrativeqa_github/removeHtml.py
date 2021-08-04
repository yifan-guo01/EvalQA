from bs4 import BeautifulSoup
import os
import string
import sys
import glob

data_files = sorted(glob.glob("tmp/*.txt"))
print(len(data_files))

for file in data_files:    
  text = ''  
  with open(file,'r',encoding='utf8') as f:
    text = f.read()
    print('file', file ,', len', len(text))
  if text.startswith('<html>'):
      print('it is html format, file name:', file)
      
      soup = BeautifulSoup(text)
      lines = soup.get_text().split('\n')
      cleantext = ''
      for line in lines:
          line = line.strip(" ")
          if len(line) != 0:
            cleantext += ' ' + line + '\n'
      with open(file+'.convert','w',encoding='utf8') as f:
          f.write(cleantext + "\n")
      
