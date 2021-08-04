dataset: https://hotpotqa.github.io/
dev directory is used to save content and questions 
answer directory is the answers from hotpot_dev_distractor_v1.json
output directory is used to save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run the commands as below:
python hotpot_evaluate_v1.py

It will compare doctalk's predictions with the answers under answer directory
then the final stats and score will be saved into output, file name is score_hotpot.txt

