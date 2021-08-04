dataset: https://github.com/ashkankzme/biased_textrank
biased_textrank_git directory is git clone code from https://github.com/ashkankzme/biased_textrank
biased_textrank_git/data/liar/clean_test.json is for test set.
biased_textrank_git/data/liar/clean_val.json is for validate set
under test and val directory:
 - dev directory is used to save articles and questions from clean_*.json
    - articles are retrieve from "statements" in clean_*.json
    - questions are retrieve from "claim" in clean_*.json
 - answer directory is used to save answer from clean_*.json
    - answers are retrieve from "new_justification" in clean_*.json

output directory is used to save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run one of the commands as below:
  - python evaluate.py test 
  - python evaluate.py val
  - python evaluate.py test2
  - python evaluate.py val2
It will compare doctalk's predictions with the answers under answer directory
then the final stats and score will be saved into output, file name is score_textrank.txt
