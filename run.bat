@echo off
call conda activate ml
python %~dp0\kcpredict\kcpredict.py
pause
call conda deactivate
