# GPT-2 Dictionary

This repo aims to provide a general codebase for conducting dictionary-learning-based mechanistic interpretability research on Language Models (LMs). It powers a configurable pipeline for training and evaluating GPT-2 dictionaries, and provides a set of tools (mainly a React-based webpage) for analyzing and visualizing the learned dictionaries.

The design of the pipeline (including the configuration and some training detail) is highly inspired by the [mats_sae_training
](https://github.com/jbloomAus/mats_sae_training) project. We thank the authors for their great work.

## Getting Started with Mechanistic Interpretability and Dictionary Learning

If you are new to the concept of mechanistic interpretability and dictionary learning, we recommend you to start from the following paper:

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382)
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600)

Furthermore, to dive deeper into the inner activations of LMs, it's recommended to get familiar with the [TransformerLens](https://github.com/neelnanda-io/TransformerLens/tree/main) library.

## Installation

Currently, the codebase use [pip](https://pip.pypa.io/en/stable/) to manage the dependencies. To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

If you want to use the visualization tools, you also need to install the required packages for the frontend, which uses [bun](https://bun.sh/) for dependency management. Follow the instructions on the website to install it, and then run the following command:

```bash
cd ui
bun install
```

It's worth noting that `bun` is not well-supported on Windows, so you may need to use WSL or other Linux-based solutions to run the frontend, or consider using a different package manager, such as `pnpm` or `yarn`.

## Project Structure Overview

The project is organized as follows:

- **core:** The core codebase for training and evaluating dictionaries.
- **ui:** React-based frontend for visualizing the learned dictionaries and their features.
- **server:** A simple FastAPI server for serving the dictionaries to the frontend.
- **tests:** Unit tests for the core codebase.
- **notebooks:** Jupyter notebooks for analyzing the learned dictionaries.
- **results:** By default, results of training and analysis will be saved in this directory.

## Training a Dictionary

We use configurations powered by dataclasses to manage the training process. It's recommended to start by copying the training example to a `exp` folder:

```bash
mkdir exp
cp run_train_example.py exp/my_train_exp.py
```

Then, modify the configuration file to fit your needs. The configurations specify the dataset, the model, the training hyperparameters, the logging and checkpointing settings, etc. The example configuration file is well-documented, so you can refer to it for more details.

After that, start the training process by running the following command:

```bash
python exp/my_exp.py
```

Make sure the experiment script is launched from the root directory of the project, so that the module can be correctly resolved by Python.

## Analyzing and Visualizing the Learned Dictionary

After training the dictionary, you can use the provided tools to analyze and visualize the learned dictionary. First, you can analyze the feature activations using the provided analyzing example:

```bash
cp run_analyze_example.py exp/my_analyze_exp.py
```

Modify the configuration file to fit your needs (especially the checkpoint path), and then run the following command:

```bash
python exp/my_analyze_exp.py
```

The analysis results will be saved in the `results/<exp_name>/analysis/<analysis_name>` directory. `notebooks/observe_feature_activation.ipynb` demonstrates how to use the analysis results to observe the feature activation of the learned dictionary.

Furthermore, you can use the provided visualization tools to visualize the learned dictionary. First, start the FastAPI server by running the following command:

```bash
uvicorn server.app:app --port 24577
# You may want to modify some environmental settings in server/.env.example to server/.env, and run with these environmental variables:
# uvicorn server.app:app --port 24577 --env-file server/.env
```

Then, copy the `ui/.env.example` file to `ui/.env` and modify the `VITE_BACKEND_URL` to fit your server settings (by default, it's `http://localhost:24577`), and start the frontend by running the following command:

```bash
cd ui
bun dev --port 24576
```

That's it! You can now go to `http://localhost:24576` to visualize the learned dictionary and its features.

## Development

We highly welcome contributions to this project. If you have any questions or suggestions, feel free to open an issue or a pull request. We are looking forward to hearing from you!

TODO: Add development guidelines

## References

TODO: Add references
