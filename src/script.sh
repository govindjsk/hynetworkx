for data in email-Enrondo  for mode in structural temporal  do    for iter in 0 1    do      file="./logs/$data.$mode.$iter"      command="python3.6 experimenter.py -d $data -s $mode -i $iter 1> $file.log 2> $file.error &"      echo "$command"      eval "$command"    done  donedone
