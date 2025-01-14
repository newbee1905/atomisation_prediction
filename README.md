# Predicting Atomisation Energy

## Libraries

- rapids or scikit-learn
- numpy
- pandas
- ase
- dscribe

## Results

|    | name                                     |           r2 |     rmse |   acc |
|---:|:-----------------------------------------|-------------:|---------:|------:|
| 33 | random_forest_sk_soap_padding_with_zeros |  0.367954    | 0.295038 | 0.763 |
|  6 | random_forest_ewald_sum                  |  0.206219    | 0.330639 | 0.734 |
|  3 | random_forest_sine                       |  0.189943    | 0.334012 | 0.742 |
| 31 | random_forest_soap__outer                |  0.173629    | 0.337358 | 0.721 |
|  0 | random_forest_coulomb                    |  0.169525    | 0.338195 | 0.725 |
| 32 | random_forest_soap__inner                |  0.145579    | 0.343036 | 0.72  |
|  1 | ridge_coulomb                            |  0.0799973   | 0.355958 | 0.72  |
|  7 | ridge_ewald_sum                          |  0.0748135   | 0.356959 | 0.72  |
|  4 | ridge_sine                               |  0.0252319   | 0.366399 | 0.721 |
| 18 | ridge_soap_truncated_expansion           |  0.00315834  | 0.370525 | 0.706 |
|  8 | lasso_ewald_sum                          | -2.54299e-05 | 0.371116 | 0.706 |
| 25 | ridge_mbtr_truncated_expansion           | -2.54299e-05 | 0.371116 | 0.706 |
| 21 | lasso_acsf_truncated_expansion           | -2.54299e-05 | 0.371116 | 0.706 |
| 23 | lasso_soap_truncated_expansion           | -2.54299e-05 | 0.371116 | 0.706 |
| 28 | ridge_lmbtr_truncated_expansion          | -2.54299e-05 | 0.371116 | 0.706 |
| 26 | lasso_mbtr_truncated_expansion           | -2.54299e-05 | 0.371116 | 0.706 |
| 29 | lasso_lmbtr_truncated_expansion          | -2.54299e-05 | 0.371116 | 0.706 |
|  5 | lasso_sine                               | -2.54299e-05 | 0.371116 | 0.706 |
|  2 | lasso_coulomb                            | -2.54299e-05 | 0.371116 | 0.706 |
| 27 | random_forest_lmbtr_truncated_expansion  | -5.89734e-05 | 0.371122 | 0.706 |
| 24 | random_forest_mbtr_truncated_expansion   | -5.89734e-05 | 0.371122 | 0.706 |
| 20 | ridge_acsf_truncated_expansion           | -0.00199134  | 0.371481 | 0.706 |
| 22 | random_forest_soap_truncated_expansion   | -0.0281701   | 0.376302 | 0.706 |
| 19 | random_forest_acsf_truncated_expansion   | -0.0528477   | 0.380791 | 0.685 |
|  9 | random_forest_coulomb_full_ds            | -0.275328    | 0.419097 | 0.711 |
| 13 | ridge_sine_full_ds                       | -0.275815    | 0.419177 | 0.706 |
| 30 | ridge_sk_soap_padding_with_zeros         | -0.278094    | 0.419552 | 0.686 |
| 12 | random_forest_sine_full_ds               | -0.372454    | 0.434763 | 0.706 |
| 16 | ridge_ewald_sum_full_ds                  | -0.399819    | 0.439076 | 0.524 |
| 15 | random_forest_ewald_sum_full_ds          | -0.402421    | 0.439484 | 0.706 |
| 10 | ridge_coulomb_full_ds                    | -0.490486    | 0.453073 | 0.702 |
| 11 | lasso_coulomb_full_ds                    | -2.66149     | 0.710122 | 0.706 |
| 17 | lasso_ewald_sum_full_ds                  | -2.66149     | 0.710122 | 0.706 |
| 14 | lasso_sine_full_ds                       | -2.66149     | 0.710122 | 0.706 |
