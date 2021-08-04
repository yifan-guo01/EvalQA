dataset: https://github.com/deepmind/narrativeqa
narrativeqa_github directory is git clone code from https://github.com/deepmind/narrativeqa
dev directory is used to save articles and questions
  - articles are retrieve from narrativeqa_github/third_party/wikipedia/summaries.csv
  - questions are retrieve from narrativeqa_github/qaps.csv
output directory is used to save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run the commands as below:
python evaluate.py

It will compare doctalk's predictions with the answers under answer directory
then the final stats and score will be saved into output, file name is score_Narrativeqa.txt
