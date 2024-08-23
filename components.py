import solara
from reacton import ipyvuetify as v

import pandas as pd


@solara.component
def TextPiece(text: str, tooltip: str, color: str = '#1953ac'):
    with solara.Div(style="display: inline;"):
        with solara.Div(
            style={
                "display": "inline",
                "padding": "10px",
                "border-right": "3px solid white",
                "line-height": "3em",
                "font-family": "courier",
                "background-color": f"{color}",
                "color": "white",
                "position": "relative",
                "font-weight": "bold"
            },
        ):
            with solara.Tooltip(tooltip=tooltip, color=color):
                solara.Text(text=text, style={"font-size": "1.2em", "color": "white"})


@solara.component
def OutputGeneral(df: pd.DataFrame, probability: float, perplexity: float):
    with solara.Card():
        with solara.Div(style="display: inline;"):
            for row in df.itertuples(index=False):
                TextPiece(text=row.text, tooltip=f'Probability: {row.prob}', color=row.color)
            with solara.VBox():
                TextPiece(text=f'Probability: {probability:.3f}', tooltip='Overall probability', color='green')
                TextPiece(text=f'Perplexity: {perplexity:.3f}', tooltip='Perplexity', color='gray')


@solara.component
def OutputClassifier(answer: str, probability: float, color: str):
    with solara.Card():
        with solara.VBox():
            TextPiece(text=f"Classification: {answer}", tooltip=f'Probability: {probability:.3f}', color=color)
            TextPiece(text=f'Probability: {probability:.3f}', tooltip='Overall probability', color=color)


@solara.component
def OutputRAG(bool_answer: bool, justification: str, answer: str, probability: float, overall_probability: float, perplexity: float, color: str):
    with solara.Card():
        with solara.VBox():
            with solara.HBox():
                TextPiece(text=f"Sufficient context: {bool_answer}", tooltip=f'Probability: {probability:.3f}', color=color)
                TextPiece(text=f'Probability: {probability:.3f}', tooltip=f'Probability: {probability:.3f}', color=color)
            solara.Markdown('')
            HTMLTEXT(header='Justification', text=justification, text_color='white', background_color='#638EC9', font_size='18px')
            solara.Markdown('')
            HTMLTEXT(header='Answer', text=answer, text_color='white', background_color='#638EC9', font_size='18px')
            solara.Markdown('')
            TextPiece(text=f'Overall Probability: {overall_probability:.3f}', tooltip='Overall probability', color=color)
            TextPiece(text=f'Perplexity: {perplexity:.3f}', tooltip='Perplexity', color='gray')


@solara.component
def TabPage(system_message: str = None, user_message: str = None, callback=None, disabled: bool = False):

    system_message = solara.use_reactive(system_message)
    user_message = solara.use_reactive(user_message)

    with solara.Div():
        with solara.Card():
            solara.Markdown("### Evaluate the response of the model based on the logprobs")
            solara.InputText(label='System message', value=system_message, disabled=True)
            solara.InputText(label='User message', value=user_message)
            with solara.Row():
                solara.Button("Submit", icon_name="mdi-play-circle-outline", color="green", on_click=lambda: callback(system_message.value, user_message.value), disabled=disabled or callback.pending)
                solara.ProgressLinear(callback.pending, color='green')
                if callback.pending:
                    solara.SpinnerSolara(size='30px')


@solara.component
def HTMLTEXT(header: str, text: str, text_color: str = 'white', background_color: str = '#1953ac', font_size: str = '20px'):

    html = f'''
        <style>
            .text-box {{
                display: inline;
                padding: 10px;
                line-height: 3em;
                font-family: courier;
                position: relative;
                font-weight: bold;
            }}
            .text-box span {{
                font-weight: normal;
            }}
        </style>

        <div class="text-box">
            {header}: <span>{text}</span>
        </div>
    '''
    style = f'background-color: {background_color}; color: {text_color}; font-size: {font_size}'
    solara.HTML(unsafe_innerHTML=html, style=style)
