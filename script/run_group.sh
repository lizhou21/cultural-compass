#!/bin/bash


# Tabel 5
task='OLD' 
features=(
    'DataSpecific'
    'Typology5'
    'Geography'
    'Orthography'
    'Typology5_Geography'
    'PRAG'
    'OFF'
    'Culture'
)





num_leaves=16
max_depth=-1
learning_rate=0.1
n_estimators=100
min_child_samples=5

python langrank_train.py --task "$task" --features "${features[@]}" --num_leaves="$num_leaves" \
    --max_depth="$max_depth" --learning_rate="$learning_rate" --n_estimators="$n_estimators" \
    --min_child_samples="$min_child_samples"
python langrank_predict.py --task "$task" --features "${features[@]}"
