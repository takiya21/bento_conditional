#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

batch_size=("32")
in_w=("640")
lr=("0.0001")
weight_decay=("0.0001" "0.00001" "0.000001")
optim=("Adam")
seed=("0")
bottle=("8" "16" "32")
for i in ${batch_size[@]}
do
  for j in ${in_w[@]}
  do
    for k in ${lr[@]}
    do
      for l in ${weight_decay[@]}
      do
        for m in ${optim[@]}
        do
            for n in ${bottle[@]}
            do
                python closs_valid_train.py --batch_size $i --in_w $j --lr $k --weight_decay $l --optim $m --seed 0 --conditional_flg 0 --bottle $n
            done
        done
      done
    done
  done
done
