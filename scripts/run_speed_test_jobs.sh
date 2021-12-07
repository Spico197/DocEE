#!/bin/bash
set -vx

NUM_TRIGGER=$1
GPUS=$2


edag() {
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'Doc2EDAG' 'doc2edag_rep' "${GPUS}" 128

    exit
}

trans_ptpcg() {
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TransTriggerAwarePrunedCompleteGraph' 'hw_TransPTPCG' "${GPUS}" 128

    exit
}

trigger1() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R1' "${GPUS}" 128

    exit
}


trigger2() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R2' "${GPUS}" 128

    exit
}


trigger3() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R3' "${GPUS}" 128

    exit
}


trigger4() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R4' "${GPUS}" 128

    exit
}


trigger5() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_R5' "${GPUS}" 128

    exit
}


trigger_all() {
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 1
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 2
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 4
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 8
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 16
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 32
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 64
    bash scripts/run_speed_test_one.sh 'TriggerAwarePrunedCompleteGraph' 'hw_TPTCG_Rall' "${GPUS}" 128

    exit
}


case $NUM_TRIGGER in 
    1)
        trigger1
        ;;
    2)
        trigger2
        ;;
    3)
        trigger3
        ;;
    4)
        trigger4
        ;;
    5)
        trigger5
        ;;
    all)
        trigger_all
        ;;
    edag)
        edag
        ;;
    trans_ptpcg)
        trans_ptpcg
        ;;
esac
