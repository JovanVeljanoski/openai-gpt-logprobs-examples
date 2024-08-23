import dataclasses
import json
import typing

import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import openai


def get_messages(system_message: str = None, user_messages: typing.Union[list, str] = None) -> list:
    '''Format a message for the openai API.
    '''
    user_messages = user_messages if isinstance(user_messages, list) else [user_messages]
    assert isinstance(system_message, (str, type(None)))
    assert isinstance(user_messages, list)
    messages = (
        [{"role": "system", "content": '' if system_message is None else system_message}] +
        [{"role": "user", "content": msg} for msg in user_messages]
    )
    return messages


def call_openai(model: str, messages: list, max_tokens: int = 256, temperature: float = 0.0, response_format: str = None, api_key: str = None) -> dict:
    '''Call the openai API.
    '''
    client = openai.Client(api_key=api_key)
    print('Calling openai API...')

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        response_format=response_format,
    )

    return response



def get_color_from_prob(prob: float, reverse: bool = False) -> str:
    '''Get a color from a probability.
    '''
    colormap = plt.cm.get_cmap('RdYlGn')
    if reverse:
        colormap = colormap.reversed()
    norm = plt.Normalize(0.2, 1)
    color = matplotlib.colors.rgb2hex(colormap(norm(prob)))
    return color


def general_call_openai(system_message: str, user_message: str, api_key: str) -> tuple[pd.DataFrame, float, float]:
    '''Call the openai API with a general model.
    '''
    messages = get_messages(system_message=system_message, user_messages=user_message)
    response = call_openai(model='gpt-4o-mini-2024-07-18', messages=messages, api_key=api_key)

    # Overall probability
    probability = np.exp(np.mean([elem.logprob for elem in response.choices[0].logprobs.content]))
    perplexity = np.exp(-np.mean([elem.logprob for elem in response.choices[0].logprobs.content]))

    df = (
        pd.DataFrame([{'token': elem.token, 'logprob': elem.logprob} for elem in response.choices[0].logprobs.content])
        .assign(punctuation=lambda x: x.token.str.contains(r'[^\w\s]', regex=True))
        .assign(sep=lambda x: x.token.str.contains(' ').cumsum())
        .groupby(by='sep')
        .agg(text=('token', 'sum'),
             logprob=('logprob', 'sum'),
             prob=('logprob', lambda x: np.round(np.exp(x.sum()), 3)))
        .assign(color=lambda x: x['prob'].apply(get_color_from_prob))
    )

    return df, probability, perplexity


def classifier_call_openai(system_message: str, user_message: str, api_key: str) -> tuple[str, float]:
    '''Call the openai API with a classifier model.
    '''
    messages = get_messages(system_message=system_message, user_messages=user_message)
    response = call_openai(model='gpt-4o-mini-2024-07-18', messages=messages, response_format={ "type": "json_object" }, api_key=api_key)

    for row in pd.DataFrame([{'token': i.token.lower().strip(), 'logprob': i.logprob} for i in response.choices[0].logprobs.content]).itertuples(index=False):
        if row.token == 'true' or row.token == 'false':
            answer = row.token
            logprob = row.logprob
            return answer, np.exp(logprob)

    print('No answer found.')
    print(pd.DataFrame([{'token': i.token.strip(), 'logprob': i.logprob} for i in response.choices[0].logprobs.content]))
    return 'test', -1


def rag_call_openai(system_message: str, user_message: str, api_key: str) -> tuple[str, float]:
    '''Call the openai API with a RAG model.
    '''
    messages = get_messages(system_message=system_message, user_messages=user_message)
    response = call_openai(model='gpt-4o-mini-2024-07-18', messages=messages, response_format={ "type": "json_object" }, api_key=api_key)
    probablity = None
    overall_prob = np.exp(np.mean([elem.logprob for elem in response.choices[0].logprobs.content]))
    preplexity = np.exp(-np.mean([elem.logprob for elem in response.choices[0].logprobs.content]))

    for row in pd.DataFrame([{'token': i.token.lower().strip(), 'logprob': i.logprob} for i in response.choices[0].logprobs.content]).itertuples(index=False):
        if row.token == 'true' or row.token == 'false':
            probablity = np.exp(row.logprob)

    if probablity is None:
        print('No answer found.')
        print(pd.DataFrame([{'token': i.token.strip(), 'logprob': i.logprob} for i in response.choices[0].logprobs.content]))
        return 'test-bool', 'test-justification', 'test-answer', -1

    response = json.loads(response.choices[0].message.content)
    return (
        response['sufficient_context_for_answer'],
        response['justification'],
        response['answer'],
        probablity,
        overall_prob,
        preplexity
    )


@dataclasses.dataclass(frozen=True)
class ResultGlobal:
    df: pd.DataFrame
    probability: float
    perplexity: float


@dataclasses.dataclass(frozen=True)
class ResultClassifier:
    answer: str
    probability: float


@dataclasses.dataclass(frozen=True)
class ResultRAG:
    bool_answer: str
    justification: str
    answer: str
    probability: float
    overall_probability: float
    perplexity: float


@dataclasses.dataclass(frozen=True)
class KeyBox:
    key: str
    valid: bool


def check_openai_api_key(api_key: str = None) -> bool:
    '''Check the openai API key.
    '''
    try:
        client = openai.Client(api_key=api_key)
        client.models.list()
    except (openai.AuthenticationError, openai.OpenAIError):
        return False
    return True
