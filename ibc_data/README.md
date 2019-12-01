## Notes about organization of the TSV files

* __main_contrasts.tsv__ contains the description of the main contrasts estimated from the task conditions. It is organized as follows:  

	* column named as *task* - task to which conditions of the contrast belong
	* column named as *contrast* - id of the contrast

* __all_contrasts.tsv__ contains the description of all meaningful contrasts estimated from the task conditions. It is organized as follows:  

	* column named as *contrast* - id of the contrast
	* column named as *task* - task to which conditions of the contrast belong
	* column named as *positive label* - label for the main regressor of the contrast
	* column named as *negative label* - label for the reversed regressor of the contrast
	* column named as *description* - description of the contrast
	* column named as *tags* - list of cognitive components describing functional activity of the contrast
