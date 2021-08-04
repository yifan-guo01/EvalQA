dataset: https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/
paragraph is the content of dataset
article is whole article that is composited by the paragraphs
under paragraph and article  directories:
- dev is used to save content and questions 
- output is save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run one of the commands as below:
python evaluate.py paragraph
python evaluate.py article

then the final stats and score will be saved into output, file name is SQuAD_1.1_score.txt



