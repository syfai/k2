#!/usr/bin/env python3
#
# Copyright      2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
# https://gradio.app/docs/#dropdown

import logging
import os
import time
import uuid

import gradio as gr
import soundfile as sf

from model import get_pretrained_model, language_to_models

title = "# Next-gen Kaldi: Text-to-speech (TTS)"

description = """
This space shows how to convert text to speech with Next-gen Kaldi.

It is running on CPU within a docker container provided by Hugging Face.

See more information by visiting the following links:

- <https://github.com/k2-fsa/sherpa-onnx>

If you want to deploy it locally, please see
<https://k2-fsa.github.io/sherpa/>
"""

# css style is copied from
# https://huggingface.co/spaces/alphacep/asr/blob/main/app.py#L113
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""


def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown.update(choices=choices, value=choices[0])

    raise ValueError(f"Unsupported language: {language}")


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def process(language: str, repo_id: str, text: str, sid: str, speed: float):
    logging.info(f"Input text: {text}. sid: {sid}, speed: {speed}")
    sid = int(sid)
    tts = get_pretrained_model(repo_id, speed)

    start = time.time()
    audio = tts.generate(text, sid=sid)
    end = time.time()

    if len(audio.samples) == 0:
        raise ValueError(
            "Error in generating audios. Please read previous error messages."
        )

    duration = len(audio.samples) / audio.sample_rate

    elapsed_seconds = end - start
    rtf = elapsed_seconds / duration

    info = f"""
    Wave duration  : {duration:.3f} s <br/>
    Processing time: {elapsed_seconds:.3f} s <br/>
    RTF: {elapsed_seconds:.3f}/{duration:.3f} = {rtf:.3f} <br/>
    """

    logging.info(info)
    logging.info(f"\nrepo_id: {repo_id}\ntext: {text}\nsid: {sid}\nspeed: {speed}")

    filename = str(uuid.uuid4())
    filename = f"{filename}.wav"
    sf.write(
        filename,
        audio.samples,
        samplerate=audio.sample_rate,
        subtype="PCM_16",
    )

    return filename, build_html_output(info)


demo = gr.Blocks(css=css)


with demo:
    gr.Markdown(title)
    language_choices = list(language_to_models.keys())

    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )

    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )

    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )

    with gr.Tabs():
        with gr.TabItem("Please input your text"):
            input_text = gr.Textbox(
                label="Input text",
                info="Your text",
                lines=3,
                placeholder="Please input your text here",
            )

            input_sid = gr.Textbox(
                label="Speaker ID",
                info="Speaker ID",
                lines=1,
                max_lines=1,
                value="0",
                placeholder="Speaker ID. Valid only for mult-speaker model",
            )

            input_speed = gr.Slider(
                minimum=0.1,
                maximum=10,
                value=1,
                step=0.1,
                label="Speed (larger->faster; smaller->slower)",
            )

            input_button = gr.Button("Submit")

            output_audio = gr.Audio(label="Output")

            output_info = gr.HTML(label="Info")

        input_button.click(
            process,
            inputs=[
                language_radio,
                model_dropdown,
                input_text,
                input_sid,
                input_speed,
            ],
            outputs=[
                output_audio,
                output_info,
            ],
        )

    gr.Markdown(description)


def download_espeak_ng_data():
    os.sytem(
        """
    cd /tmp
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
    tar xf espeak-ng-data.tar.bz2
    """
    )


if __name__ == "__main__":
    download_espeak_ng_data()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    demo.launch()
