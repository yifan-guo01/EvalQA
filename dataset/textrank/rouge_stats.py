import rouge

def rstat(type, all_hypothesis,all_references ):
  if type == rouge:
    return rouge_rstat(all_hypothesis,all_references)
  else:
    return pyrouge_rstat(all_hypothesis,all_references)



#https://www.aclweb.org/anthology/2020.coling-main.144.pdf should use this rouge
def rouge_rstat(all_hypothesis,all_references):
    # pip install rouge
    rouge = rouge.Rouge()
    scores = rouge.get_scores(all_hypothesis, all_references)
    return scores

# pip install py-rouge
def pyrouge_rstat(all_hypothesis,all_references) :
  for aggregator in ['Individual'] : #['Avg', 'Best', 'Individual']:
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=2,
                           limit_length=False,
                           #length_limit=300,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=False)
    scores = evaluator.get_scores(all_hypothesis, all_references)
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
      yield results

def hyps_and_refs() :
  
  hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
  reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
  return (hypothesis,reference)

def go() :
  hs,ds=hyps_and_refs()
  for r in rstat(hs,ds) : print(r)

if __name__=='__main__' :
  go()
    