# -*- coding: utf-8 -*-
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

import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names
from transformers import AutoTokenizer, AutoModel

def generate_image(dense_class_vector=None, name=None, noise_seed_vector=None, truncation=0.4,
                   gan_model=None, pretrained_gan_model_name='biggan-deep-128'):
    """ Utility function to generate an image (numpy uint8 array) from either:
        - a name (string): converted in an associated ImageNet class and then
            a dense class embedding using BigGAN's internal ImageNet class embeddings.
        - a dense_class_vector (torch.Tensor with 128 elements): used as a replacement of BigGAN internal
            ImageNet class embeddings.
        
        Other args:
            - noise_seed_vector: a vector used to control the seed (seed set to the sum of the vector elements)
            - truncation: a float between 0 and 1 to control image quality/diversity tradeoff (see BigGAN paper)
            - gan_model: a BigGAN model from pytorch_pretrained_biggan library.
                If None a model is instanciated from a pretrained model name given by `pretrained_gan_model_name`
                List of possible names: https://github.com/huggingface/pytorch-pretrained-BigGAN#models
            - pretrained_gan_model_name: shortcut name of the GAN model to instantiate if no gan_model is provided. Default to 'biggan-deep-128'
    """
    seed = int(noise_seed_vector.sum().item()) if noise_seed_vector is not None else None
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1, seed=seed)
    noise_vector = torch.from_numpy(noise_vector)

    if gan_model is None:
        gan_model = BigGAN.from_pretrained(pretrained_gan_model_name)

    if name is not None:
        class_vector = one_hot_from_names([name], batch_size=1)
        class_vector = torch.from_numpy(class_vector)
        dense_class_vector = gan_model.embeddings(class_vector)
        # input_vector = torch.cat([noise_vector, gan_class_vect.unsqueeze(0)], dim=1)
        # dense_class_vector = torch.matmul(class_vector, gan.embeddings.weight.t())
    else:
        dense_class_vector = dense_class_vector.view(1, 128)

    input_vector = torch.cat([noise_vector, dense_class_vector], dim=1)

    # Generate an image
    with torch.no_grad():
        output = gan_model.generator(input_vector, truncation)
    output = output.cpu().numpy()
    output = output.transpose((0, 2, 3, 1))
    output = ((output + 1.0) / 2.0) * 256
    output.clip(0, 255, out=output)
    output = np.asarray(np.uint8(output[0]), dtype=np.uint8)
    return output


def print_image(numpy_array, legend):
    """ Utility function to print a numpy uint8 array as an image
    """
    img = Image.fromarray(numpy_array)
    plt.imshow(img)
    plt.title(legend)
    plt.show()


def text_to_image(text, mapping_model,
                  lm_model=None, lm_tokenizer=None, pretrained_lm_model_name='distilbert-base-uncased',
                  gan_model=None, pretrained_gan_model_name='biggan-deep-128',
                  truncation=0.4,
                  noise_seed=None):
    """ Utility function to plug your `mapping model` in and display an image from a text string.
        
        mapping_model should be a model that
             - take as input a sequence of hidden-states of shape (seq length, LM hidden size)
             - and output a class vector for the GAN model with GAN's hidden size elements (128 elements for BiGAN).
    """
    if lm_tokenizer is None:
        lm_tokenizer = AutoTokenizer.from_pretrained(pretrained_lm_model_name)
    if lm_model is None:
        lm_model = AutoModel.from_pretrained(pretrained_lm_model_name)

    tokens = lm_tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        lm_output = lm_model(tokens)
        lm_hidden_states = lm_output[0]  # Last layer hidden-states are the first output of Transformers' library models output tuple
        lm_hidden_states_first_example = lm_hidden_states[0]  # Keep first example in the batch - output shape (seq length, hidden size)

        ####################################
        # This is where your magic happens
        # `mapping_output` should be a torch.Tensor with 128 elements to be used as
        # a replacement of BigGAN internal ImageNet class embeddings.
        #
        mapping_output = mapping_model(lm_hidden_states_first_example)
        #
        ####################################

    # Now generate an image (a numpy array)
    numpy_image = generate_image(mapping_output,
                                 gan_model=gan_model,
                                 pretrained_gan_model_name=pretrained_gan_model_name,
                                 truncation=truncation,
                                 noise_seed_vector=tokens if noise_seed is None else noise_seed)
    print_image(numpy_image, text)
