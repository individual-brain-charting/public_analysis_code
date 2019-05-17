## Notes about organization of the TSV files

* __main_contrasts.tsv__ contains the description of the main contrasts estimated from the task conditions. It is organized as follows:  

	* column named as *task* - task to which conditions of the contrast belong
	* column named as *contrast* - id of the contrast
	* column named as *pretty name* - description of the contrast
	* column named as *left label* - label for the control condition of the contrast
	* column named as *right label* - label for the main condition of the contrast  

* __all_contrasts.tsv__ contains the description of all meaningful contrasts estimated from the task conditions. It is organized as follows:  

	* column named as *contrast* - id of the contrast
	* column named as *task* - task to which conditions of the contrast belong
	* column named as *pretty name* - description of the contrast
	* column named as *left label* - label for the control condition of the contrast
	* column named as *right label* - label for the main condition of the contrast
	* all remaining columns refer to the occurence of the cognitive component, stated in the column header, for the corresponding contrast
