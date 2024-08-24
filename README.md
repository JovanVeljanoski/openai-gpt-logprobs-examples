---
title: openai-gpt-logprobs-examples
emoji: 🚀
colorFrom: red
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

### Simple examples of using `logprobs` from outputs of OpenAI GPT models

This repo contains some simple examples of how to use, display and use `logprobs`
as part of the output of GPT models from OpenAI.

It runs as a simple [solara](https://solara.dev/) web-application with 3 tabs:


1. Get the `logprobs` per word, from a single reponse from a LLM model, coloured by their probability. This way you can see the possible variations in the answer you are getting.

2. A binary classification example. The reply is a `True` or `False`. The probability is extracted from the `logprob` of the `True/False` token only. Can be used to "tune" your classifier based on this probability, much like you can do with a classical ML classifier.

3. A simple RAG example. It combines the elements of both examples above.

### Links

- [GitHub repo](https://github.com/JovanVeljanoski/openai-gpt-logprobs-examples)
- [Huggingface space](https://huggingface.co/spaces/Jovan31/openai-gpt-logprobs-examples)


### To install / use locally

Clone this repo locally:

```bash
git clone git@github.com:JovanVeljanoski/openai-gpt-logprobs-examples.git
```

Install the dependencies, preferrably in a new (virtual) environment:

```bash
pip install -r requirements.txt
```

Launch the app

```bash
solara run app.py
```
