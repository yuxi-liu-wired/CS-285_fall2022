#!/bin/bash

file=cs285/report/reported_data.txt
rm -rf reported_data
mkdir -p reported_data
while read line; do 
    echo "Copying data/${line}..."
    cp -r "data/${line}" reported_data; 
done < "${file}"

mkdir -p submit
rm -rf submit/cs285
rm -rf submit/run_logs
rm -f submit/submit.zip
cp -r cs285 submit/cs285
cp -r reported_data submit/run_logs