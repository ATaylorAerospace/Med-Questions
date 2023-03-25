# Med-Questions
Syntax Instructions For LLM Training for Dissimiliar Medical Question Pairs
--
annotations_creators:
- expert-generated
language_creators:
- other
language:
- en
license:
- unknown
multilinguality:
- multilingual
size_categories:
- 1K<n<60K
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- semantic-similarity-classification
pretty_name: MedQuestionsPairs
dataset_info:
  features:
  - name: dr_id
    dtype: int32
  - name: indication_1
    dtype: string
  - name: question_2
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': 0
          '1': 1
  splits:
  - name: train
    num_bytes: 3601842
    num_examples: 29046
  download_size: 4759948
  dataset_size: 948736
