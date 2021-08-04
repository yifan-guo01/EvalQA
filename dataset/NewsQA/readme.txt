dataset: https://github.com/Maluuba/newsqa
dev directory is used to save content and questions 
answer directory is the answers from combined-newsqa-data-v1.json
output directory is used to save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run the commands as below:
python evaluate.py

It will compare doctalk's predictions with the answers under answer directory
then the final stats and score will be saved into output, file name is score_NewsQA.txt

