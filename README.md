# PubMedCLIP in Medical Visual Question Answering

This repository includes PubMedCLIP, the fine-tuned version of CLIP with ROCO image--caption pairs. We also provide the pipelines for encorporating PubMedCLIP as the alternative pre-trained visual encoder in [MEVF](https://arxiv.org/abs/1909.11867) and [QCR](https://dl.acm.org/doi/abs/10.1145/3394171.3413761?casa_token=E_IrwKfXPEMAAAAA:IC1Epmj0HbdWYzZWUfPpjbBJuMuL-iTdGbe1kVr5UQ4iVvfTgN_mgDBBEjyhqNBzRanKKlzyVQ) medical visual question answering pipelines. Our experiments illustrate that PubMedCLIP results in up tp 3% improvement in the medical visual question answering.

## Citation
If you use this work in academic publication, please cite the [paper](https://aclanthology.org/2023.findings-eacl.88/) by [Sedigheh Eslami](https://github.com/SarahESL), [Christoph Meinel](https://hpi.de/en/meinel/chair/prof-dr-ch-meinel.html), and [Gerard de Melo](http://gerard.demelo.org/).

BibTeX entry:
```
@inproceedings{eslami2023pubmedclip,
  title={PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain?},
  author={Eslami, Sedigheh and Meinel, Christoph and De Melo, Gerard},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2023},
  pages={1151--1163},
  year={2023}
}
```
