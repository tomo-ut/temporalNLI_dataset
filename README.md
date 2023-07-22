# Jamp: Controlled Japanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models

## About Jamp and this repository

**Jamp** is the Japanese temporal inference benchmark. This repository consists of templates, test data, and training data. The test data and training data both include tokenized and non-tokenized data. Tokenized data contains `wakati` in the file names. The training data contains both non-split data that includes all problems and split data following the methodology described in the paper. Files containing `template`, `time format`, or `time span` in their names are split based on **tense fragment**, **time format**, or **time span**, respectively.


## Citation

If you use this dataset in any published research, please cite the following:

- Tomoki Sugimoto, Yasumasa Onoe, and Hitomi Yanaka. 2023. Jamp: Controlled Japanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop), pages 57–68, Toronto, Canada. Association for Computational Linguistics.

```
@inproceedings{sugimoto-etal-2023-jamp,
    title = "Jamp: Controlled {J}apanese Temporal Inference Dataset for Evaluating Generalization Capacity of Language Models",
    author = "Sugimoto, Tomoki  and
      Onoe, Yasumasa  and
      Yanaka, Hitomi",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-srw.8",
    pages = "57--68",
}
```

## Contact

For questions and usage issues, please contact sugimoto.tomoki@is.s.u-tokyo.ac.jp