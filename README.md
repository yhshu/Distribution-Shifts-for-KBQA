# Distribution Shifts for KBQA

Official code for the paper "Distribution Shifts Are Bottlenecks: Extensive Evaluation for
Grounding Language Models to Knowledge Bases". 

[[arXiv](https://arxiv.org/pdf/2309.08345.pdf)] [Proceedings] [[Hugging Face Datasets](https://huggingface.co/datasets/yhshu/TIARA-GAIN/tree/main)]

This repo contains a data augmentation method named **G**raph se**A**rch and quest**I**on generatio**N** (GAIN).
GAIN could be used to augment any neural KBQA models.

## Citation

```
@article{shu2023data,
      title={Distribution Shifts Are Bottlenecks: Extensive Evaluation for Grounding Language Models to Knowledge Bases}, 
      author={Yiheng Shu and Zhiwei Yu},
      year={2023},
      eprint={2309.08345},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Setup

### Freebase Setup

Please follow [Freebase Setup](https://github.com/dki-lab/Freebase-Setup) to set up a Virtuoso service. Note that at least 30G RAM and 53G+ disk space is needed for Freebase
Virtuoso. The download may take some time. The default port of this service is `localhost:3001`. If you change the port of Virtuoso service, please also modify the Freebase port setting in `utils/config.py`.

The working dir for following commands is `src`.

## 1. Graph Search

Graph search for logical form:

```shell
python algorithm/graph_query/logical_form_search.py --domain synthetic --output_dir ../dataset/question_generation
```

Graph search for triple:

```shell
python algorithm/graph_query/triple_search.py --output_dir ../dataset/question_generation
```

## 2. Training Question Generation Model & 3. Verbalization

If the QG models have been trained, the training will be skipped and verbalization will be performed.
In this step, you can directly use our implementation or modify the code to train a verbalizer on any KBQA datasets with logical form / triple annotations.

Training QG model for logical form:

```shell
python algorithm/question_generation/logical_form_question_generation.py --model_dir ../model/logical_form_question_generation
```

Training QG model for triple:

```shell
python algorithm/question_generation/triple_question_generation.py --model_dir ../model/triple_question_generation
```

## 4. Pre-training on Synthetic Data

How to use the synthetic data and which KBQA model to use depends on your choice. In this paper, the synthetic dataset is used to pre-train a KBQA model and the model is fine-tuned
on different KBQA datasets, respectively.

## 5. Evaluation

We modify the official evaluation scripts of GrailQA and GraphQuestions for paraphrase adaptation, i.e., `utils/statistics/grailqa_evaluate.py`
and `utils/statistics/graphq_evaluate.py`.

To evaluate your QA results with `utils/statistics/graphq_evaluate.py`, you may need to generate a result template via `utils/statistics/graphq_evaluate_template.py`. The template is based on [this result format](https://github.com/ysu1989/GraphQuestions/tree/master/freebase13/results).

## Data Resources

Datasets and retrieval results using TIARA and TIARA + GAIN can be found at [Hugging Face Datasets](https://huggingface.co/datasets/yhshu/TIARA-GAIN/tree/main).

### Datasets

- [GrailQA](https://dki-lab.github.io/GrailQA/)
- [GraphQuestions Freebase 2013 version](https://github.com/ysu1989/GraphQuestions)
- [GraphQuestions Freebase 2015 version](https://github.com/dki-lab/ArcaneQA/tree/main/data)
- [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763)
- [SimpleQuestions - Balanced](https://github.com/wudapeng268/KBQA-Adapter/tree/master/Data/SQB)

### Retrieval Results with GAIN

- GrailQA exemplary logical form retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf) + GAIN)
- GrailQA schema retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf) + GAIN)
- GraphQuestions exemplary logical form retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf))
- GraphQuestions exemplary logical form retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf) + GAIN)
- GraphQuestions schema retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf))
- GraphQuestions schema retrieval ([TIARA](https://arxiv.org/pdf/2210.12925.pdf) + GAIN)