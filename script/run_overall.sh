#!/bin/bash
# results in Table 2

task='OLD'


features=(
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG' # LangRank + PRAG
    'DataSpecific_TTR_Orthography_Typology5_Geography_Culture' # LangRank + Culture
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_Culture' # LangRank + PRAG + Culture
    'DataSpecific_TTR_Orthography_Typology5_Geography_OFF' # LangRank + Culture + OFF
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_OFF' # LangRank + PRAG
    'DataSpecific_TTR_Orthography_Typology5_Geography_Culture_OFF' # LangRank + Culture
    'DataSpecific_TTR_Orthography_Typology5_Geography_PRAG_Culture_OFF' # LangRank + PRAG + Culture

    'Learned_PRAG' # MTVEC + PRAG
    'Learned_Culture' # MTVEC + Culture
    'Learned_PRAG_Culture' # MTVEC + PRAG + Culture
    'Learned_OFF' # MTVEC + OFF
    'Learned_PRAG_OFF' # MTVEC + PRAG
    'Learned_Culture_OFF' # MTVEC + Culture
    'Learned_PRAG_Culture_OFF' # MTVEC + PRAG + Culture

    'Colex'
    'Colex_PRAG' # MTVEC + PRAG
    'Colex_Culture' # MTVEC + Culture
    'Colex_PRAG_Culture' # MTVEC + PRAG + Culture
    'Colex_OFF'
    'Colex_PRAG_OFF' # MTVEC + PRAG
    'Colex_Culture_OFF' # MTVEC + Culture
    'Colex_PRAG_Culture_OFF' # MTVEC + PRAG + Culture

)


num_leaves=16
max_depth=-1
learning_rate=0.1
n_estimators=100
min_child_samples=5

# python langrank_train.py --task "$task" --features "${features[@]}" --num_leaves="$num_leaves" \
#     --max_depth="$max_depth" --learning_rate="$learning_rate" --n_estimators="$n_estimators" \
#     --min_child_samples="$min_child_samples"
python langrank_predict.py --task "$task" --features "${features[@]}"
