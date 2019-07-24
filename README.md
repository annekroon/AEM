# AEM

This repo attempts to build 'The Amsterdam word embedding model': A news-specific word embedding model trained on Dutch journalistic content.

---
This repo contains the following elements:

- Training of Word Embeddings Models:
    - We use Word2Vec to train sets of models on smaller and larger samples with different parameter settings
- Evaluation of these models:
	- Intrinsic evaluation (i.e., syntatic and semantic accuracy of the models)
	- Extrinsic evaluation (i.e., performance of the models in downstream tasks)

In doing so, we compare the here-trained models with pre-trained word embedding models on Dutch corpora.

---

## Python scripts:

### run_classifier.py

This script will execute the classification class in `lib/`: It will return accuracy scores for classification with SGD and ExtraTrees with or without embeddings.

To run:

```
python3 run_classifier.py  --word_embedding_path ../repos/embedding_models/w2v_model_nr_7_window_10_size_300_negsample_15 --word_embedding_sample_size large  --type_vectorizer Tfidf --data_path data/dataset_vermeer.pkl --output output/output

```

## Directory

- `lib/`: Modules used in python scripts: Classification
- `output/`: Default output directory
- `helpers/`: small scripts to get info on training samples

## Directory

The current study tests the quality of classifiers w/wo embedding vectorizers on the following data:

Vermeer, S.: [Dutch News Classifier] (https://figshare.com/articles/A_supervised_machine_learning_method_to_classify_Dutch-language_news_items/7314896/1)
This datasets classifies Dutch news in four topics

Buscher, Vliegenthart & De Vrees: [Policy Issues Classifier] https://www.google.com/search?q=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&oq=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&aqs=chrome..69i57.688j0j7&sourceid=chrome&ie=UTF-8
This paper tests a classifer of 18 topics on Dutch news

We thank the authors of these papers for sharing their data. If there are any issues with the way we handle the data/ suggestions: please contact us.
