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
__main_contrasts.tsv__ contains only the contrasts isolating effects-of-interest. Most of the IBC tasks refer to categorical designs. Therefore, "main contrasts" are defined in terms of one of the following options: (1) "active condition *vs.* control condition"; (2) "control condition *vs.* baseline"; and, sometimes, (3) "active condition *vs.* baseline". For the few tasks that follow a parametric design, a "main contrast" can also be considered as "the parametric effect of the constant effect in the active condition *vs.* baseline". Importantly, main contrasts within tasks are linearly independent between them and, consequently, this also stands true across tasks.

__all_contrasts.tsv__ contains all possible contrasts that can be extracted from the task paradigm. 

Note: Reverse contrasts are not listed in the 'main_contrasts.tsv' and, in most of the cases, contrasts formed by elementary conditions. Yet, if a main contrast is composed by an active condition and a control condition, we also include the contrast formed by the control condition *vs.* baseline in the 'main_contrasts.tsv'.
