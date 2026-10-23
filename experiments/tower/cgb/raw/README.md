---
license: mit
extra_gated_prompt: >-
  You agree to not use the dataset to conduct experiments that cause harm to
  human subjects. Please note that the data in this dataset may be subject to
  other agreements. Before using the data, be sure to read the relevant
  agreements carefully to ensure compliant use. Video copyrights belong to the
  original video creators or platforms and are for academic research use only.
task_categories:
- visual-question-answering
extra_gated_fields:
  Name: text
  Company/Organization: text
  Country: text
  E-Mail: text
modalities:
- Video
- Text
configs:
- config_name: cg-bench
  data_files: cgbench.json
- config_name: cg-bench-mini
  data_files: cgbench_mini.json
language:
- en
size_categories:
- 10K<n<100K
---

# CG-Bench

Project Website: https://cg-bench.github.io/leaderboard/  
GitHub Repository: https://github.com/CG-Bench/CG-Bench (includes running code)

## Summary

We introduce CG-Bench, a groundbreaking benchmark for clue-grounded question answering in long videos, addressing the limitations of existing benchmarks that focus primarily on short videos and rely on multiple-choice questions (MCQs). These limitations allow models to answer by elimination rather than genuine understanding. CG-Bench enhances evaluation credibility by requiring models to retrieve relevant clues for questions. It includes 1,219 manually curated videos across 14 primary, 171 secondary, and 638 tertiary categories, making it the largest benchmark for long video analysis. With 12,129 QA pairs in perception, reasoning, and hallucination question types, CG-Bench introduces innovative clue-based evaluation methods: clue-grounded white box and black box evaluations, ensuring answers are based on correct video understanding. Evaluations of various MLLMs reveal significant performance gaps in long video comprehension, especially between open-source and commercial models. We aim for CG-Bench to drive the development of more reliable and capable MLLMs for long video understanding. All annotations and video data will be publicly released.

<div align="center">
  <img src="./asset/summary.jpg" width="100%" alt="CG-Bench Summary"/>
</div>

## Leaderboard

Please visit our [project page](https://cg-bench.github.io/leaderboard/) for the latest leaderboard.

## Benchmark Statistics

**Video Meta:** Our dataset comprises a total of 1219 videos with multiple multimodal information, including vision, audio, and subtitles. The duration of the videos varies between 10 and 80 minutes. Notably, videos that last between 20 and 30 minutes are the most prevalent. This selection process is manual, based on content relevance, which mirrors real-world duration distributions and highlights a long-tail effect for longer videos. As illustrated in Figure 2, each video is classified using a three-tiered tagging system that succinctly encapsulates its content and assigns it to fundamental categories. The primary classification is augmented by a secondary layer of 171 tags and a tertiary layer consisting of 638 tags. This multi-level tagging mechanism guarantees a broad diversity of data content. For a more detailed classification of tags, please consult the supplementary materials.

**Question Meta:** We annotate it with high-quality question-answer-clue (QAC) triplets. To ensure question diversity, we first establish a taxonomy with three main types: Perception, Reasoning, and Hallucination. As shown in Figure 3, Perception and Reasoning questions are further divided into 10 and 14 subcategories, respectively, while Hallucination questions combine elements of both perception and reasoning. Annotators are instructed to include negative options to create a multiple-choice QA format, facilitating straightforward and cost-effective assessments. To minimize expression loss, annotators use their native language during the annotation process. Each video requires between 6 to 15 QAC triplets, depending on its duration.

<div align="center">
  <img src="./asset/benchmark_stat.jpg" width="100%" alt="Benchmark Statistics"/>
</div>

## Benchmark Comparison

CG-Bench is characterized by its diverse features, allowing it to be compared with three distinct types of benchmarks, as depicted in the three sections of Table: Question Clue Grounding, Short-Video QA, and Long-Video QA benchmarks.

**Question Grounding:** \textcolor{blue}{For the question clue grounding benchmarks, NextGQA, Ego4D-NLQ, MultiHop-EgoQA, E.T. Bench, and RexTime are primarily centered around action and egocentric domains. Their videos are sampled from academic datasets.} In comparison, the question clue grounding part of CG-Bench, CG-Bench-QG, stands out with the highest number of videos and the longest average length, the diversity of which fosters a broad spectrum of question-grounding queries.

**Short-Video Question Answering:** Furthermore, we transform QAC triplets to our novel Short-Video QA benchmark, termed CG-Bench-Clue. When contrasted with prior short video benchmarks such as TempCompass, MVBench and MMBench-Video, our CG-Bench-Clue emerges as the *largest*, *held-out*, *open-domain* and *multimodal* Short-Video QA benchmark.

**Long-Video Question Answering:** As for the Long-Video QA benchmark, CG-Bench excels in the number of videos, length, quantity of questions, and annotation quality. Owing to our clue interval annotations, CG-Bench further facilitates reliable evaluations for long videos and open-ended evaluations with clue assistance, a feature that sets it apart from existing long video benchmarks like Video-MME and MLVU.

<div align="center">
  <img src="./asset/benchmark_comparison.jpg" width="100%" alt="Benchmark Comparison"/>
</div>

## Experiments Results

<div align="center">
  <img src="./asset/experiments.jpg" width="100%" alt="Experiments Results"/>
</div>

## Citation

```bibtex
@misc{chen2024cgbench,
      title={CG-Bench: Clue-grounded Question Answering Benchmark for Long Video Understanding}, 
      author={Guo Chen and Yicheng Liu and Yifei Huang and Yuping He and Baoqi Pei and Jilan Xu and Yali Wang and Tong Lu and Limin Wang},
      year={2024},
      eprint={2412.12075},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```