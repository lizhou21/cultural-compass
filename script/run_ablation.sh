#!/bin/bash
task='OLD'


features=(
    'Learned_PRAG_Culture_OFF' # MTVEC + PRAG + Culture
    'Learned_PRAG_noPDI_OFF'
    'Learned_PRAG_noIDV_OFF'
    'Learned_PRAG_noMAS_OFF'
    'Learned_PRAG_noUAI_OFF'
    'Learned_PRAG_noLTO_OFF'
    'Learned_PRAG_noIVR_OFF'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_Culture'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noPDI'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noIDV'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noMAS'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noUAI'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noLTO'
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_noIVR'
    'Colex_Culture_OFF' # MTVEC + Culture
    'Colex_noPDI_OFF'
    'Colex_noIDV_OFF'
    'Colex_noMAS_OFF'
    'Colex_noUAI_OFF'
    'Colex_noLTO_OFF'
    'Colex_noIVR_OFF'
    
)

# features=('OFF')
num_leaves=16
max_depth=-1
learning_rate=0.1
n_estimators=100
min_child_samples=5

# python langrank_train.py --task "$task" --features "${features[@]}" --num_leaves="$num_leaves" \
#     --max_depth="$max_depth" --learning_rate="$learning_rate" --n_estimators="$n_estimators" \
#     --min_child_samples="$min_child_samples"
python langrank_predict.py --task "$task" --features "${features[@]}"
