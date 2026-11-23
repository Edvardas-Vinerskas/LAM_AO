#!/bin/bash
# all FATMOSS first-stage runs: perfect-WFS gain sweeps + PWFS run_first_stage_loop (gain 0.5), in parallel
cd /home/evinerskas/LAM_OOPAO
export PYTHONPATH=.:PAPYRIIS_2stage_CNN_RL MPLBACKEND=Agg
PY="$HOME/ao/bin/python PAPYRIIS_2stage_CNN_RL/first_stage_fatmoss.py"
LOG=PAPYRIIS_2stage_CNN_RL/logs_fatmoss
mkdir -p $LOG

th() { OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 "${@:2}"; }

for B in 200 400 1000; do
    th 4 $PY pwfs    $B > $LOG/pwfs_B$B.log    2>&1 &
    th 2 $PY perfect $B > $LOG/perfect_B$B.log 2>&1 &
done
(
    th 4 $PY project 0 > $LOG/project_B0.log 2>&1 &&
    { th 4 $PY pwfs 0 > $LOG/pwfs_B0.log 2>&1 & th 2 $PY perfect 0 > $LOG/perfect_B0.log 2>&1 & wait; }
) &
wait
echo ALL DONE
