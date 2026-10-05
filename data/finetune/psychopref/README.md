---
license: cc-by-nc-4.0
task_categories:
- text-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: prefID
    dtype: int64
  - name: question
    dtype: string
  - name: chosen
    dtype: string
  - name: rejected
    dtype: string
  - name: chosen_model
    dtype: string
  - name: rejected_model
    dtype: string
  - name: chosen_empathy_rating
    dtype: int64
  - name: chosen_relevance_rating
    dtype: int64
  - name: chosen_clarity_rating
    dtype: int64
  - name: chosen_safety_rating
    dtype: int64
  - name: chosen_exploration_rating
    dtype: int64
  - name: chosen_autonomy_rating
    dtype: int64
  - name: chosen_staging_rating
    dtype: int64
  - name: rejected_empathy_rating
    dtype: int64
  - name: rejected_relevance_rating
    dtype: int64
  - name: rejected_clarity_rating
    dtype: int64
  - name: rejected_safety_rating
    dtype: int64
  - name: rejected_exploration_rating
    dtype: int64
  - name: rejected_autonomy_rating
    dtype: int64
  - name: rejected_staging_rating
    dtype: int64
  splits:
  - name: train
    num_bytes: 112988097
    num_examples: 34329
  - name: test
    num_bytes: 7715540
    num_examples: 2324
  download_size: 41375576
  dataset_size: 120703637
---

Dataset from the paper [Preference Learning Unlocks LLMs' Psycho-Counseling Skills](https://huggingface.co/papers/2502.19731).