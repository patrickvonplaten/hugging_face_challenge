# 🤗Hugging Face ML researcher/engineer code exercise - Funky mutli-modal version

Restitution for Hugging Face's Funky multi-modal coding exercise. The chosen tensor calculus / deep learning library is PyTorch (v1.7).

## Description

### Dependency management

All dependencies required to run the code are listed in the `requirements.txt` file.

### `text2img` package

The `text2img` directory contains a Python package with utility code for the exercise. it has 3 sub modules: 
- `text2img.data`, in which are implemented utility functions to build the datasets
- `text2img.models`, in which are implemented the Deep Learning models
- `text2img.optimization`, where a custom bi-objective loss is implemented.

### Notebooks

The restitution itself consists in 4 jupyter notebooks, under the `notebooks` directory:
- `1_generate_sentence_dataset.ipynb` where we build a dataset of sentences related to ImageNet classes
-  `2_generate_representation_mapping_dataset.ipynb` where we use this sentece dataset to build a dataset of "source" and "target" representations to learn the repreentation mapper
-  `3_simple_linear_model.ipynb` where we implement a basic linear representation mapping using PCA and the orthogonal procrustes problem
-  `4_auto_encoder.ipynb` where we implement and learn an auto-encoder-based representation mapping

All notebooks should be runnable end to end, if that's not the case feel free to reach out.

### Final word

That's it, I hope you'll enjoy my work as much as I enjoyed doing it !
