# Data Report

## Shape
1200 rows × 25 columns

## Null Percentage per Column
```
patient_id            0.0
age                   0.0
sex                   0.0
bmi                   0.0
iop_od                0.0
iop_os                0.0
cup_disc_ratio        0.0
va_od                 0.0
va_os                 0.0
mean_deviation_od    10.0
mean_deviation_os    10.0
pattern_sd            8.0
rnfl_superior         7.0
rnfl_inferior         7.0
rnfl_average          0.0
hba1c                12.0
systolic_bp           0.0
diastolic_bp          0.0
diabetes              0.0
hypertension          0.0
family_history        0.0
treatment             0.0
eye_color             0.0
ethnicity             0.0
disease_severity      0.0
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
Normal               529
Suspect              403
Mild Glaucoma        217
Moderate Glaucoma     46
Severe Glaucoma        5
```
Imbalance ratio (max/min): 105.80
