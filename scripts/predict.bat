@echo off
if "%2" == "" goto args_count_wrong
if "%3" == "" goto args_count_ok

:args_count_wrong
echo Syntax: predict.bat existing_input_csv_file desired_output_csv_file
exit /b 1

:args_count_ok
echo Input file: %1%
echo Output file: %2%

docker pull piotrtomaszewski/shape_prediction
docker run -t --rm -v %CD%:/data piotrtomaszewski/shape_prediction --input /data/%1  --output /data/%2 -v
