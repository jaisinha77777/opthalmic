# Data Report

## Shape
4000 rows × 25 columns

## Null Percentage per Column
```
patient_id            0.00
age                   0.00
sex                   0.00
bmi                   0.00
iop_od                0.00
iop_os                0.00
cup_disc_ratio        0.00
va_od                 0.00
va_os                 0.00
mean_deviation_od    10.02
mean_deviation_os    10.10
pattern_sd            8.43
rnfl_superior         6.85
rnfl_inferior         6.88
rnfl_average          0.00
hba1c                11.85
systolic_bp           0.00
diastolic_bp          0.00
diabetes              0.00
hypertension          0.00
family_history        0.00
treatment             0.00
eye_color             0.00
ethnicity             0.00
disease_severity      0.00
```

## dtypes
```
patient_id               str
age                  float64
sex                    int64
bmi                  float64
iop_od               float64
iop_os               float64
cup_disc_ratio       float64
va_od                float64
va_os                float64
mean_deviation_od    float64
mean_deviation_os    float64
pattern_sd           float64
rnfl_superior        float64
rnfl_inferior        float64
rnfl_average         float64
hba1c                float64
systolic_bp          float64
diastolic_bp         float64
diabetes               int64
hypertension           int64
family_history         int64
treatment                str
eye_color                str
ethnicity                str
disease_severity         str
```

## Column Classification
- Numerical: ['age', 'bmi', 'iop_od', 'iop_os', 'cup_disc_ratio', 'va_od', 'va_os', 'mean_deviation_od', 'mean_deviation_os', 'pattern_sd', 'rnfl_superior', 'rnfl_inferior', 'rnfl_average', 'hba1c', 'systolic_bp', 'diastolic_bp']
- Categorical: ['treatment', 'eye_color', 'ethnicity', 'disease_severity']
- Binary: ['sex', 'diabetes', 'hypertension', 'family_history']
- High Cardinality (>50 unique): ['patient_id']
- ID columns: ['patient_id']

## Target Column
`disease_severity`

## Class Distribution
```
disease_severity
Suspect              1798
Moderate Glaucoma     871
Severe Glaucoma       525
Normal                470
Mild Glaucoma         336
```
Imbalance ratio (max/min): 5.35
