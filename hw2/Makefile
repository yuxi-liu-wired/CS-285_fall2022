SHELL=/bin/bash

submit.zip:
	mkdir -p submit
	rm -rf submit/cs285
	rm -rf submit/run_logs
	rm -f submit.zip
	cp -r cs285 submit/cs285
	cp -r data submit/run_logs

file := cs285/report/reported_data.txt
collect_data:
	# can't get even this to work...
	# rm -rf reported_data
	# mkdir -p reported_data
	# while read line; do cp -r $${line} reported_data; done < "${file}"
