for %%I in (.) do set CurrDirName=%%~nxI
CALL conda activate %CurrDirName%
jupyter notebook
