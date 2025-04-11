# **(D1`s)Performances of ECD,FT,and RCCA. A dash("-") indicates that the methid does not address the respective problem**
| Method        | ECD Precision | ECD Recall | ECD F1-score | FT Precision | FT Recall | FT F1-score | RCCA Top1 | RCCA Top3 | RCCA AVG@5 |
|---------------|-----------|------------|--------------|----------|-----------|-------------|------------|-------------|---------------|
| **SCEAM**     | 1.000     | 1.000      | 1.000        | 0.964    | 0.964     | 0.964       | 0.775      | 0.900       | 0.975         |
| **SCWarm**    | 0.900     | 0.857      | 0.878        | -        | -         | -           | -          | -           | -             |
| **Kontrast**  | 0.925     | 0.860      | 0.892        | -        | -         | -           | -          | -           | -             |
| **Lumos**     | 0.926     | 0.841      | 0.881        | -        | -         | -           | -          | -           | -             |
| **Funnel**    | 0.875     | 0.875      | 0.875        | -        | -         | -           | -          | -           | -             |
| **Ganaldf**   | 0.900     | 0.818      | 0.857        | -        | -         | -           | -          | -           | -             |
| **MircroCBR** | -         | -          | -            | 0.461    | 0.461     | 0.461       | -          | -           | -             |
| **PDResponse**| -         | -          | -            | -        | -         | -           | 0.025      | 0.100       | 0.150         |
# **(D2`s)Performances of ECD,FT,and RCCA. A dash("-") indicates that the methid does not address the respective problem**
| Method        | ECD Precision | ECD Recall | ECD F1-score | FT Precision | FT Recall | FT F1-score | RCCA Top1 | RCCA Top3 | RCCA AVG@5 |
|---------------|---------------|------------|--------------|--------------|-----------|-------------|------------|-----------|------------|
| **SCEAM**     | 1.000         | 0.891      | 0.942        | 0.861        | 0.870     | 0.865       | 0.879      | 0.932     | 0.932      |
| **SCWarm**    | 0.929         | 0.891      | 0.909        | -            | -         | -           | -          | -         | -          |
| **Kontrast**  | 0.891         | 0.876      | 0.884        | -            | -         | -           | -          | -         | -          |
| **Lumos**     | 0.874         | 0.837      | 0.855        | -            | -         | -           | -          | -         | -          |
| **Funnel**    | 0.858         | 0.867      | 0.863        | -            | -         | -           | -          | -         | -          |
| **Ganaldf**   | 0.880         | 0.880      | 0.880        | 0.418        | 0.420     | 0.414       | -          | -         | -          |
| **MircroCBR** | -             | -          | -            | -            | -         | -           | 0.067      | 0.307     | 0.507      |
| **PDResponse**| -             | -          | -            | -            | -         | -           | 0.025      | 0.100     | 0.150      |
# **Processing Time for Each Change Case (in seconds)**
| Method    | ECD | FT | RCCA | D1     | D2     |
|-----------|-----|----|------|--------|--------|
| SCELM     | ✓   | ✓  | ✓    | 6.357  | 6.977  |
| SCWarm    | ✓   |   |     | 4.375  | 2.189  |
| Kontrast  | ✓   |   |     | 6.331  | 2.829  |
| Lumos     | ✓   |   |     | 4.375  | 1.928  |
| Funnel    | ✓   |   |     | 8.875  | 3.928  |
| Gandalf   | ✓   |   |     | 3.605  | 2.920  |
| MicroCBR  |    | ✓  |     | 47.937 | 19.279 |
| PDiagnose |    |   | ✓    | 0.357  | 0.179  |
# **The Evaluation Results of Ablation Study**
| Stage | Evaluation | D1: SCELM | D1: A1  | D1: A2  | D2: SCELM | D2: A1  | D2: A2  |
|-------|------------|-----------|--------|--------|-----------|--------|--------|
| ECD   | Precision  | 1.000     | 1.000  | 1.000  | 1.000     | 0.764  | 1.000  |
| ECD   | Recall     | 1.000     | 1.000  | 1.000  | 0.891     | 0.979  | 0.943  |
| ECD   | F1-score   | 1.000     | 1.000  | 1.000  | 0.942     | 0.858  | 0.971  |
| FT    | Precision  | 0.964     | 0.864  | 0.864  | 0.870     | 0.825  | 0.838  |
| FT    | Recall     | 0.964     | 0.926  | 0.929  | 0.865     | 0.648  | 0.690  |
| FT    | F1-score   | 0.964     | 0.895  | 0.895  | 0.861     | 0.659  | 0.723  |
| RCCA  | Top1       | 0.775     | 0.000  | 0.100  | 0.879     | 0.147  | 0.542  |
| RCCA  | Top3       | 0.900     | 0.000  | 0.120  | 0.932     | 0.158  | 0.542  |
| RCCA  | AVG@5      | 0.975     | 0.000  | 0.120  | 0.932     | 0.163  | 0.542  |


