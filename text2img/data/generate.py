# -*- coding: utf-8 -*-
# COPY-PASTED from rpepare_data.py script

# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os

import nltk
import torch
from nltk.corpus import wordnet as wn
from pytorch_pretrained_biggan.utils import IMAGENET
from tqdm import tqdm
from transformers import AutoTokenizer

nltk.download('wordnet')

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

PATTERNS = [
    'i saw a <WORD>.',
]

def generate_dataset(patterns=None, lm_tokenizer=None, pretrained_lm_model_name='distilbert-base-uncased'):
    """ Utility function to generate a dataset of string sequence associated to each imagenet class
        Args:
            - patterns: a list of string to use as patterns in which <WORD> will be replaced by
                a word related to each ImageNet class (e.g. dog, fish...).
                Example of pattern: 'i saw a <WORD>.' will be converted in 'i saw a dog' for the ImageNet class associated to dog.
            - lm_tokenizer: a tokenizer from Transformers library.
                If None a tokenizer is instanciated from a pretrained model vocabulary given by `pretrained_lm_model_name`
                List of possible names: https://huggingface.co/transformers/pretrained_models.html
            - pretrained_lm_model_name: shortcut name of the pretrained language model tokenizer to instantiate if no
                lm_tokenizer is provided. Default to 'distilbert-base-uncased'
        
        Returns: tuple with:
            - tokens_list: list (batch) of list (sequence) of int (language model voc indices)
            - labels_list: list (batch) of int (imagenet class labels)
            - texts_list: list (batch) of string (text before tokenization)
    """
    if lm_tokenizer is None:
        lm_tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_model_name)
    if patterns is None:
        patterns = PATTERNS

    # The IMAGENET dict provided in pytorch_pretrained_biggan is
    # a dict of {wordnet.synset_offset: ImageNet class index}
    # For more informations on ImageNet classes relation with wordnet:
    # see http://www.image-net.org/download-API
    # Let's inverse the dict and link it with wordnet synsets:
    class_to_synset = dict((v, wn.synset_from_pos_and_offset('n', k)) for k, v in IMAGENET.items())

    # %%
    # Some classes in ImageNet are associated to complexe and rare words like breeds of dog
    # To make it simpler for us in the first step, we'll filter the words
    # to associate a more common word to each class or remove the class if we can't find a
    # common word or if all the common words associated to the class are already taken.
    # A common word here is a word that is in Bert tokenizer vocabulary.
    logger.info('Building a list of simple words associated to ImageNet classes')
    words_dataset = {}
    all_words = set()
    for i, synset in tqdm(class_to_synset.items()):
        current_synset = synset
        while current_synset:
            for lemma in current_synset.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name in all_words:
                    continue  # Word is already assigned
                if lm_tokenizer.convert_tokens_to_ids(name) != lm_tokenizer.unk_token_id:
                    # Word is in Bert tokenizer vocabulary
                    words_dataset[i] = name
                    all_words.add(name)
                    current_synset = False # We're good
                    break
            if current_synset and current_synset.hypernyms():
                current_synset = current_synset.hypernyms()[0]
            else:
                current_synset = False

    # Now let's build a simple sentence for each ImageNet category
    # to use as input to our language model
    logger.info('Building a list of sentences associated to each selected ImageNet class')
    # Here we only provide one simple pattern but better ways
    # to build a diverse dataset could be used.
    examples_dataset = {}
    for i, word in tqdm(words_dataset.items()):
        examples_dataset[i] = [pat.replace('<WORD>', word) for pat in patterns]

    return examples_dataset


def tokenize_dataset(dataset, lm_tokenizer=None, pretrained_lm_model_name='distilbert-base-uncased'):
    """ Utility function to tokenize a dataset
        Args:
            - dataset: a dict {ImageNet class index: list of examples as string related to the class}
            - lm_tokenizer: a tokenizer from Transformers library.
                If None a tokenizer is instanciated from a pretrained model vocabulary given by `pretrained_lm_model_name`
                List of possible names: https://huggingface.co/transformers/pretrained_models.html
            - pretrained_lm_model_name: shortcut name of the pretrained language model tokenizer to instantiate if no
                lm_tokenizer is provided. Default to 'distilbert-base-uncased'
        
        Returns: tuple with:
            - tokens_list: list (batch) of list (sequence) of int (language model voc indices)
            - labels_list: list (batch) of int (imagenet class labels)
            - texts_list: list (batch) of string (text before tokenization)
    """
    if lm_tokenizer is None:
        lm_tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_model_name)

    labels_list = []
    tokens_list = []
    texts_list = []
    for class_index, examples in tqdm(dataset.items()):
        examples_tokens = [lm_tokenizer.encode(ex, add_special_tokens=True) for ex in examples]
        tokens_list.extend(examples_tokens)
        labels_list += [class_index] * len(examples_tokens)
        texts_list += examples

    return (tokens_list, labels_list, texts_list)


def generate_simple_dataset(args):
    """ Utility function to generate a very simple dataset of string sequence associated to each imagenet class.
        The sentence are tokenized and gathered in a torch.Tensor of shape (num of examples, input sequence)
        Args:
            - args.pretrained_lm_model_name: shortcut name of a tokenizer from Transformers library.
                List of possible names: https://huggingface.co/transformers/pretrained_models.html
            - args.output_dir: Output directory to save the encoded dataset to.
        
        Returns: Nothing (save dataset to output_dir)
    """
    lm_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm_model_name)

    dataset = generate_dataset(patterns=PATTERNS,
                               lm_tokenizer=lm_tokenizer)

    tokenized_output = tokenize_dataset(dataset,
                                        lm_tokenizer=lm_tokenizer)
    tokens_list, labels_list, texts_list = tokenized_output

    # We are cheating a bit here:
    # since all our input have the same length (only one simple pattern) we can just build
    # a big tensor from them without having to worry about padding.
    # Update this if you built a more diverse input dataset.
    labels_tensor = torch.tensor(labels_list)
    tokens_tensor = torch.tensor(tokens_list)

    # Save encoded inputs, labels and associated texts
    torch.save(labels_tensor, os.path.join(args.output_dir, 'labels_tensor.bin'))
    torch.save(tokens_tensor, os.path.join(args.output_dir, 'tokens_tensor.bin'))
    with open(os.path.join(args.output_dir, 'input_texts.txt'), 'w') as f:
        for item in texts_list:
            f.write("%s\n" % item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_lm_model_name", default='distilbert-base-uncased', type=str,
                        help="Pretrained language model vocabulary to use for pre-processing. "
                             "See full list at https://huggingface.co/transformers/pretrained_models.html")
    parser.add_argument("--output_dir", default='./data/', type=str,
                        help="Output directory to save the generated dataset to. "
                             "We save labels and tokens as torch.Tensors and the associated sentences as a text file "
                             "with one example per line.")

    args = parser.parse_args()
    generate_simple_dataset(args)
