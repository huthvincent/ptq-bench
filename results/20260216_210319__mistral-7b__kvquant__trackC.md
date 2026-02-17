# 实验结果: mistral-7b + kvquant (Track C)

## 运行信息

- **运行时间**: 2026-02-16T21:03:19.359297
- **脚本**: `scripts/run_one.py`
- **完整 CLI 参数**:
  ```
  scripts/run_one.py --model mistral-7b --method kvquant --track C
  ```

## 数据集

- **校准数据集**: N/A
- **校准样本数**: N/A
- **校准序列长度**: N/A
- **PPL 评测数据集**: 
- **lm-eval 任务**: 

## 量化参数

- **方法**: kvquant
- **赛道**: Track C
- **kv**: {"key_bits": 2, "value_bits": 2, "num_outliers": 1, "residual_length": 32}
- **Seed**: 42

## PPL 结果

| 数据集 | PPL | NLL |
|--------|-----|-----|
| wikitext2 | 4.7909 | 1.566708 |
| longbench | 8.2641 | 2.111919 |

## lm-eval 任务结果

| 任务 | 指标 | 分数 |
|------|------|------|
| hellaswag | alias | hellaswag |
| hellaswag | acc,none | 0.6115 |
| hellaswag | acc_stderr,none | 0.0049 |
| hellaswag | acc_norm,none | 0.8063 |
| hellaswag | acc_norm_stderr,none | 0.0039 |
| mmlu | acc,none | 0.5959 |
| mmlu | acc_stderr,none | 0.0039 |
| mmlu | alias | mmlu |
| mmlu_humanities | acc,none | 0.5345 |
| mmlu_humanities | acc_stderr,none | 0.0067 |
| mmlu_humanities | alias |  - humanities |
| mmlu_formal_logic | alias |   - formal_logic |
| mmlu_formal_logic | acc,none | 0.373 |
| mmlu_formal_logic | acc_stderr,none | 0.0433 |
| mmlu_high_school_european_history | alias |   - high_school_european_history |
| mmlu_high_school_european_history | acc,none | 0.7394 |
| mmlu_high_school_european_history | acc_stderr,none | 0.0343 |
| mmlu_high_school_us_history | alias |   - high_school_us_history |
| mmlu_high_school_us_history | acc,none | 0.8137 |
| mmlu_high_school_us_history | acc_stderr,none | 0.0273 |
| mmlu_high_school_world_history | alias |   - high_school_world_history |
| mmlu_high_school_world_history | acc,none | 0.7764 |
| mmlu_high_school_world_history | acc_stderr,none | 0.0271 |
| mmlu_international_law | alias |   - international_law |
| mmlu_international_law | acc,none | 0.7355 |
| mmlu_international_law | acc_stderr,none | 0.0403 |
| mmlu_jurisprudence | alias |   - jurisprudence |
| mmlu_jurisprudence | acc,none | 0.6852 |
| mmlu_jurisprudence | acc_stderr,none | 0.0449 |
| mmlu_logical_fallacies | alias |   - logical_fallacies |
| mmlu_logical_fallacies | acc,none | 0.7485 |
| mmlu_logical_fallacies | acc_stderr,none | 0.0341 |
| mmlu_moral_disputes | alias |   - moral_disputes |
| mmlu_moral_disputes | acc,none | 0.659 |
| mmlu_moral_disputes | acc_stderr,none | 0.0255 |
| mmlu_moral_scenarios | alias |   - moral_scenarios |
| mmlu_moral_scenarios | acc,none | 0.2391 |
| mmlu_moral_scenarios | acc_stderr,none | 0.0143 |
| mmlu_philosophy | alias |   - philosophy |
| mmlu_philosophy | acc,none | 0.6881 |
| mmlu_philosophy | acc_stderr,none | 0.0263 |
| mmlu_prehistory | alias |   - prehistory |
| mmlu_prehistory | acc,none | 0.679 |
| mmlu_prehistory | acc_stderr,none | 0.026 |
| mmlu_professional_law | alias |   - professional_law |
| mmlu_professional_law | acc,none | 0.4511 |
| mmlu_professional_law | acc_stderr,none | 0.0127 |
| mmlu_world_religions | alias |   - world_religions |
| mmlu_world_religions | acc,none | 0.8363 |
| mmlu_world_religions | acc_stderr,none | 0.0284 |
| mmlu_other | acc,none | 0.6836 |
| mmlu_other | acc_stderr,none | 0.0081 |
| mmlu_other | alias |  - other |
| mmlu_business_ethics | alias |   - business_ethics |
| mmlu_business_ethics | acc,none | 0.6 |
| mmlu_business_ethics | acc_stderr,none | 0.0492 |
| mmlu_clinical_knowledge | alias |   - clinical_knowledge |
| mmlu_clinical_knowledge | acc,none | 0.6906 |
| mmlu_clinical_knowledge | acc_stderr,none | 0.0285 |
| mmlu_college_medicine | alias |   - college_medicine |
| mmlu_college_medicine | acc,none | 0.5954 |
| mmlu_college_medicine | acc_stderr,none | 0.0374 |
| mmlu_global_facts | alias |   - global_facts |
| mmlu_global_facts | acc,none | 0.39 |
| mmlu_global_facts | acc_stderr,none | 0.049 |
| mmlu_human_aging | alias |   - human_aging |
| mmlu_human_aging | acc,none | 0.6726 |
| mmlu_human_aging | acc_stderr,none | 0.0315 |
| mmlu_management | alias |   - management |
| mmlu_management | acc,none | 0.8252 |
| mmlu_management | acc_stderr,none | 0.0376 |
| mmlu_marketing | alias |   - marketing |
| mmlu_marketing | acc,none | 0.8376 |
| mmlu_marketing | acc_stderr,none | 0.0242 |
| mmlu_medical_genetics | alias |   - medical_genetics |
| mmlu_medical_genetics | acc,none | 0.69 |
| mmlu_medical_genetics | acc_stderr,none | 0.0465 |
| mmlu_miscellaneous | alias |   - miscellaneous |
| mmlu_miscellaneous | acc,none | 0.7957 |
| mmlu_miscellaneous | acc_stderr,none | 0.0144 |
| mmlu_nutrition | alias |   - nutrition |
| mmlu_nutrition | acc,none | 0.6961 |
| mmlu_nutrition | acc_stderr,none | 0.0263 |
| mmlu_professional_accounting | alias |   - professional_accounting |
| mmlu_professional_accounting | acc,none | 0.4752 |
| mmlu_professional_accounting | acc_stderr,none | 0.0298 |
| mmlu_professional_medicine | alias |   - professional_medicine |
| mmlu_professional_medicine | acc,none | 0.6801 |
| mmlu_professional_medicine | acc_stderr,none | 0.0283 |
| mmlu_virology | alias |   - virology |
| mmlu_virology | acc,none | 0.506 |
| mmlu_virology | acc_stderr,none | 0.0389 |
| mmlu_social_sciences | acc,none | 0.6952 |
| mmlu_social_sciences | acc_stderr,none | 0.0081 |
| mmlu_social_sciences | alias |  - social sciences |
| mmlu_econometrics | alias |   - econometrics |
| mmlu_econometrics | acc,none | 0.4035 |
| mmlu_econometrics | acc_stderr,none | 0.0462 |
| mmlu_high_school_geography | alias |   - high_school_geography |
| mmlu_high_school_geography | acc,none | 0.7828 |
| mmlu_high_school_geography | acc_stderr,none | 0.0294 |
| mmlu_high_school_government_and_politics | alias |   - high_school_government_and_politics |
| mmlu_high_school_government_and_politics | acc,none | 0.8497 |
| mmlu_high_school_government_and_politics | acc_stderr,none | 0.0258 |
| mmlu_high_school_macroeconomics | alias |   - high_school_macroeconomics |
| mmlu_high_school_macroeconomics | acc,none | 0.5974 |
| mmlu_high_school_macroeconomics | acc_stderr,none | 0.0249 |
| mmlu_high_school_microeconomics | alias |   - high_school_microeconomics |
| mmlu_high_school_microeconomics | acc,none | 0.6303 |
| mmlu_high_school_microeconomics | acc_stderr,none | 0.0314 |
| mmlu_high_school_psychology | alias |   - high_school_psychology |
| mmlu_high_school_psychology | acc,none | 0.7945 |
| mmlu_high_school_psychology | acc_stderr,none | 0.0173 |
| mmlu_human_sexuality | alias |   - human_sexuality |
| mmlu_human_sexuality | acc,none | 0.7252 |
| mmlu_human_sexuality | acc_stderr,none | 0.0392 |
| mmlu_professional_psychology | alias |   - professional_psychology |
| mmlu_professional_psychology | acc,none | 0.6062 |
| mmlu_professional_psychology | acc_stderr,none | 0.0198 |
| mmlu_public_relations | alias |   - public_relations |
| mmlu_public_relations | acc,none | 0.6273 |
| mmlu_public_relations | acc_stderr,none | 0.0463 |
| mmlu_security_studies | alias |   - security_studies |
| mmlu_security_studies | acc,none | 0.702 |
| mmlu_security_studies | acc_stderr,none | 0.0293 |
| mmlu_sociology | alias |   - sociology |
| mmlu_sociology | acc,none | 0.8408 |
| mmlu_sociology | acc_stderr,none | 0.0259 |
| mmlu_us_foreign_policy | alias |   - us_foreign_policy |
| mmlu_us_foreign_policy | acc,none | 0.82 |
| mmlu_us_foreign_policy | acc_stderr,none | 0.0386 |
| mmlu_stem | acc,none | 0.504 |
| mmlu_stem | acc_stderr,none | 0.0086 |
| mmlu_stem | alias |  - stem |
| mmlu_abstract_algebra | alias |   - abstract_algebra |
| mmlu_abstract_algebra | acc,none | 0.31 |
| mmlu_abstract_algebra | acc_stderr,none | 0.0465 |
| mmlu_anatomy | alias |   - anatomy |
| mmlu_anatomy | acc,none | 0.563 |
| mmlu_anatomy | acc_stderr,none | 0.0428 |
| mmlu_astronomy | alias |   - astronomy |
| mmlu_astronomy | acc,none | 0.6053 |
| mmlu_astronomy | acc_stderr,none | 0.0398 |
| mmlu_college_biology | alias |   - college_biology |
| mmlu_college_biology | acc,none | 0.6944 |
| mmlu_college_biology | acc_stderr,none | 0.0385 |
| mmlu_college_chemistry | alias |   - college_chemistry |
| mmlu_college_chemistry | acc,none | 0.44 |
| mmlu_college_chemistry | acc_stderr,none | 0.0499 |
| mmlu_college_computer_science | alias |   - college_computer_science |
| mmlu_college_computer_science | acc,none | 0.54 |
| mmlu_college_computer_science | acc_stderr,none | 0.0501 |
| mmlu_college_mathematics | alias |   - college_mathematics |
| mmlu_college_mathematics | acc,none | 0.38 |
| mmlu_college_mathematics | acc_stderr,none | 0.0488 |
| mmlu_college_physics | alias |   - college_physics |
| mmlu_college_physics | acc,none | 0.3922 |
| mmlu_college_physics | acc_stderr,none | 0.0486 |
| mmlu_computer_security | alias |   - computer_security |
| mmlu_computer_security | acc,none | 0.73 |
| mmlu_computer_security | acc_stderr,none | 0.0446 |
| mmlu_conceptual_physics | alias |   - conceptual_physics |
| mmlu_conceptual_physics | acc,none | 0.5191 |
| mmlu_conceptual_physics | acc_stderr,none | 0.0327 |
| mmlu_electrical_engineering | alias |   - electrical_engineering |
| mmlu_electrical_engineering | acc,none | 0.5793 |
| mmlu_electrical_engineering | acc_stderr,none | 0.0411 |
| mmlu_elementary_mathematics | alias |   - elementary_mathematics |
| mmlu_elementary_mathematics | acc,none | 0.3783 |
| mmlu_elementary_mathematics | acc_stderr,none | 0.025 |
| mmlu_high_school_biology | alias |   - high_school_biology |
| mmlu_high_school_biology | acc,none | 0.7645 |
| mmlu_high_school_biology | acc_stderr,none | 0.0241 |
| mmlu_high_school_chemistry | alias |   - high_school_chemistry |
| mmlu_high_school_chemistry | acc,none | 0.4828 |
| mmlu_high_school_chemistry | acc_stderr,none | 0.0352 |
| mmlu_high_school_computer_science | alias |   - high_school_computer_science |
| mmlu_high_school_computer_science | acc,none | 0.62 |
| mmlu_high_school_computer_science | acc_stderr,none | 0.0488 |
| mmlu_high_school_mathematics | alias |   - high_school_mathematics |
| mmlu_high_school_mathematics | acc,none | 0.3222 |
| mmlu_high_school_mathematics | acc_stderr,none | 0.0285 |
| mmlu_high_school_physics | alias |   - high_school_physics |
| mmlu_high_school_physics | acc,none | 0.3444 |
| mmlu_high_school_physics | acc_stderr,none | 0.0388 |
| mmlu_high_school_statistics | alias |   - high_school_statistics |
| mmlu_high_school_statistics | acc,none | 0.4815 |
| mmlu_high_school_statistics | acc_stderr,none | 0.0341 |
| mmlu_machine_learning | alias |   - machine_learning |
| mmlu_machine_learning | acc,none | 0.4643 |
| mmlu_machine_learning | acc_stderr,none | 0.0473 |
| winogrande | alias | winogrande |
| winogrande | acc,none | 0.7435 |
| winogrande | acc_stderr,none | 0.0123 |

**平均准确率**: 0.6131

## 系统指标

- **VRAM 峰值**: 16453.8 MB
- **评测总耗时**: 806.6 秒

## 运行环境

- **时间**: 2026-02-16T21:03:19.333164
- **主机**: sev-cxl
- **OS**: Linux 6.11.0-snp-host-68799c0277b2
- **Python**: 3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]
- **Git**: `8a4fa89e` (main) ⚠️ (有未提交修改)
- **GPU 0**: NVIDIA H200 NVL (143771 MB)
- **Driver**: 580.95.05
- **CUDA**: 12.8
- **cuDNN**: 91002

### 关键包版本

- `torch`: 2.10.0+cu128
- `transformers`: 5.1.0
- `datasets`: 4.5.0
- `accelerate`: 1.12.0
- `lm-eval`: 0.4.11
- `safetensors`: 0.7.0
- `tokenizers`: 0.22.2
- `scipy`: 1.17.0
- `numpy`: 2.4.2
