- Sample data used for training:

  - A total of 250 variables were used.
  - The data cannot be disclosed due to security reasons related to medical data.


    | Sex | Age | Wt | ...| HR_rss | Asthma | med_Dopamine_HCl |
    |----------|----------|----------|----------|----------|----------|----------|
    | 1 | 49 | 68.9 | ...| 817.0600 |1| 0.0 | 
    | 0 |	84 | 46.5	| ...| 390.2422 |0| 5.0 |
    | 1 | 81 | 67.6 | ...| 376.1330 |0| 0.0 | 


- Mutual Information Data

  - The mutual information between independent variables and the dependent variable was calculated and stored.
 
      | col_name | mutual_info | 
    |----------|----------|
    | Sex | 0.000003 |
    | Age |	0.022626 |
    | Wt | 0.022626 |
    | ... | ... |
    | HR_rss | 0.2134 |
    | Asthma | 0.00048 |
    | med_Dopamine_HCl | 0.047271 |
