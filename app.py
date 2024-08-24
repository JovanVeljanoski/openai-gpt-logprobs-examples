import os

import solara
import solara.lab

from components import (
    OutputClassifier,
    OutputGeneral,
    OutputRAG,
    TabPage,
)

import keukenhof

from utils import (
    check_openai_api_key,
    classifier_call_openai,
    general_call_openai,
    get_color_from_prob,
    KeyBox,
    rag_call_openai,
    ResultClassifier,
    ResultGlobal,
    ResultRAG,
)


# Config
keybox = solara.reactive(KeyBox(key=os.environ.get('OPENAI_API_KEY', ''), valid=check_openai_api_key(api_key=os.environ.get('OPENAI_API_KEY', ''))))


# Default messages
# Global
system_message_global = "Give short witty sarcastic answers."
user_message_global = "What is the meaning of life?"

# Classifier
system_message_classifier = '''
Classify wether the main point of the text is about sports or not. \n
Reply only with True or False. Reply in a JSON format only. \n
Examples: \n

```
Reggie Miller is the best 3-pt shooter in NBA history. -> {"answer": True} \n
The capital of France is Paris. -> {"answer": False} \n
Mike Tyson spent time in prison. -> {"answer": False} \n
```
'''
user_message_classifier = "Novak Djokovic likes to drink smoothies for breakfast."

# RAG
system_message_rag = f'''Use the following article to answer questions about Keukenhof. \n
First determine whether the question is answerable based on the article. This can be only answered as `True` or `False` \n
Then provide justification about why you think that is \n
Finally provide the answer. \n
Do not make stuff up. \n

Always reply in JSON format. Use the following schema: `{{"sufficient_context_for_answer": bool , "justification": str , "answer": str}}` \n
```
{keukenhof.text}
```
'''
user_message_rag = "What is the name of the castle that Keukenhof is situated on?"


@solara.lab.task
def general_call(system_message, user_message):
    df, probability, perplexity = general_call_openai(system_message=system_message, user_message=user_message, api_key=keybox.value.key)
    return ResultGlobal(df=df, probability=probability, perplexity=perplexity)


@solara.lab.task
def classifier_call(system_message, user_message):
    answer, probability = classifier_call_openai(system_message=system_message, user_message=user_message, api_key=keybox.value.key)
    return ResultClassifier(answer=answer, probability=probability)

@solara.lab.task
def rag_call(system_message, user_message):
    bool_answer, justification, answer, probability, overall_probability, perplexity = rag_call_openai(system_message=system_message, user_message=user_message, api_key=keybox.value.key)
    return ResultRAG(bool_answer=bool_answer, justification=justification, answer=answer, probability=probability, overall_probability=overall_probability, perplexity=perplexity)

def update_keybox(api_key):
    keybox.set(KeyBox(key=api_key, valid=check_openai_api_key(api_key=api_key)))



@solara.component
def Page():
    solara.Title('OpenAI GPT Logprobs Examples')
    with solara.Column(margin=11):
        solara.Markdown("## OpenAI GPT Logprobs Examples")

        with solara.lab.Tabs(color='green'):
            with solara.lab.Tab("General"):
                TabPage(system_message=system_message_global, user_message=user_message_global, callback=general_call, disabled=keybox.value.valid is False)
                if general_call.value is not None:
                    OutputGeneral(df=general_call.value.df, probability=general_call.value.probability, perplexity=general_call.value.perplexity)


            with solara.lab.Tab("Classifier"):
                TabPage(system_message=system_message_classifier, user_message=user_message_classifier, callback=classifier_call, disabled=keybox.value.valid is False)
                if classifier_call.value is not None:
                    OutputClassifier(answer=classifier_call.value.answer, probability=classifier_call.value.probability, color=get_color_from_prob(classifier_call.value.probability))


            with solara.lab.Tab("RAG"):
                with solara.Card():
                    with solara.Details('System message', expand=False):
                            solara.Markdown(system_message_rag)
                    user_message = solara.use_reactive(user_message_rag)
                    solara.InputText(label='User message', value=user_message)
                    solara.Button("Submit", icon_name="mdi-play-circle-outline", color="green", on_click=lambda: rag_call(system_message_rag, user_message.value), disabled=keybox.value.valid is False)
                solara.ProgressLinear(rag_call.pending, color='green')
                if rag_call.value is not None:
                    OutputRAG(bool_answer=rag_call.value.bool_answer, justification=rag_call.value.justification, answer=rag_call.value.answer, probability=rag_call.value.probability, overall_probability=rag_call.value.overall_probability, perplexity=rag_call.value.perplexity, color=get_color_from_prob(rag_call.value.probability))


        with solara.Details(summary='Authentication - OpenAI API Key', expand= not keybox.value.valid):
            if keybox.value.valid:
                solara.Success('A valid API key is set.')
            else:
                solara.Warning('You need a valid OpenAI API key to use the application.')
            solara.InputText(label='OpenAI API Key', on_value=update_keybox)
