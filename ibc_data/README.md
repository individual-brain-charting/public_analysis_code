## Notes about organization of the TSV files

* __main_contrasts.tsv__ contains the description of the main contrasts estimated from the conditions. It is organized as follows:  

	* column named as *task* - id of the task
	* column named as *contrast* - id of the contrast

* __all_contrasts.tsv__ contains the description of all meaningful contrasts estimated from the conditions. It is organized as follows:  
	
	* column named as *task* - id of the task
	* column named as *contrast* - id of the contrast
	* column named as *positive label* - label for the main regressor of the contrast
	* column named as *negative label* - label for the reversed regressor of the contrast
	* column named as *description* - description of the contrast
	* column named as *tags* - list of cognitive components describing functional activity of the contrast

* __conditions.tsv__ contains the list of all independent (or elementary) contrasts. They are formed by the elementary conditions *vs.* baseline. It is organized as follows:  

	* column named as *task* - id of the task
	* column named as *contrast* - id of the contrast referring to the elementary condition
	
## Main *versus* All contrasts
__main_contrasts.tsv__ contains only the contrasts depicting the most relevant effects-of-interest, whereas __all_contrasts.tsv__ contains all possible contrasts that can be extracted from the task paradigm. Note that the reverse contrasts are not listed in the 'main_contrasts.tsv' as well as contrasts formed by elementary conditions. Yet, if a main contrast is composed by an active condition and a control condition, we also include the contrast formed by the control condition *vs.* baseline in the 'main_contrasts.tsv'.
