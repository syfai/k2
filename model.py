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

from functools import lru_cache

import sherpa_onnx
from huggingface_hub import hf_hub_download


def get_file(
    repo_id: str,
    filename: str,
    subfolder: str = ".",
) -> str:
    model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return model_filename


@lru_cache(maxsize=10)
def _get_vits_vctk(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    assert repo_id == "csukuangfj/vits-vctk"

    model = get_file(
        repo_id=repo_id,
        filename="vits-vctk.onnx",
        subfolder=".",
    )

    lexicon = get_file(
        repo_id=repo_id,
        filename="lexicon.txt",
        subfolder=".",
    )

    tokens = get_file(
        repo_id=repo_id,
        filename="tokens.txt",
        subfolder=".",
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                lexicon=lexicon,
                tokens=tokens,
                length_scale=1.0 / speed,
            ),
            provider="cpu",
            debug=True,
            num_threads=2,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def _get_vits_ljs(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    assert repo_id == "csukuangfj/vits-ljs"

    model = get_file(
        repo_id=repo_id,
        filename="vits-ljs.onnx",
        subfolder=".",
    )

    lexicon = get_file(
        repo_id=repo_id,
        filename="lexicon.txt",
        subfolder=".",
    )

    tokens = get_file(
        repo_id=repo_id,
        filename="tokens.txt",
        subfolder=".",
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                lexicon=lexicon,
                tokens=tokens,
                length_scale=1.0 / speed,
            ),
            provider="cpu",
            debug=True,
            num_threads=2,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def _get_vits_piper(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    n = len("vits-piper-")
    name = repo_id.split("/")[1][n:]

    model = get_file(
        repo_id=repo_id,
        filename=f"{name}.onnx",
        subfolder=".",
    )

    lexicon = get_file(
        repo_id=repo_id,
        filename="lexicon.txt",
        subfolder=".",
    )

    tokens = get_file(
        repo_id=repo_id,
        filename="tokens.txt",
        subfolder=".",
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                lexicon=lexicon,
                tokens=tokens,
                length_scale=1.0 / speed,
            ),
            provider="cpu",
            debug=True,
            num_threads=2,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def _get_vits_zh_aishell3(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    assert repo_id == "csukuangfj/vits-zh-aishell3"

    model = get_file(
        repo_id=repo_id,
        filename="vits-aishell3.onnx",
        subfolder=".",
    )

    lexicon = get_file(
        repo_id=repo_id,
        filename="lexicon.txt",
        subfolder=".",
    )

    tokens = get_file(
        repo_id=repo_id,
        filename="tokens.txt",
        subfolder=".",
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                lexicon=lexicon,
                tokens=tokens,
                length_scale=1.0 / speed,
            ),
            provider="cpu",
            debug=True,
            num_threads=2,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def get_pretrained_model(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    if repo_id in chinese_models:
        return chinese_models[repo_id](repo_id, speed)
    elif repo_id in english_models:
        return english_models[repo_id](repo_id, speed)
    elif repo_id in german_models:
        return german_models[repo_id](repo_id, speed)
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


chinese_models = {
    "csukuangfj/vits-zh-aishell3": _get_vits_zh_aishell3,
}

english_models = {
    "csukuangfj/vits-vctk": _get_vits_vctk,  # 109 speakers
    "csukuangfj/vits-ljs": _get_vits_ljs,
    # piper, US
    "csukuangfj/vits-piper-en_US-amy-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-amy-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-arctic-medium": _get_vits_piper,  # 18 speakers
    "csukuangfj/vits-piper-en_US-danny-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-hfc_male-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-joe-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-kathleen-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-kusal-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-l2arctic-medium": _get_vits_piper,  # 24 speakers
    "csukuangfj/vits-piper-en_US-lessac-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-lessac-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-lessac-high": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-libritts-high": _get_vits_piper,  # 904 speakers
    "csukuangfj/vits-piper-en_US-libritts_r-medium": _get_vits_piper,  # 904 speakers
    "csukuangfj/vits-piper-en_US-ryan-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-ryan-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_US-ryan-high": _get_vits_piper,
    # piper, GB
    "csukuangfj/vits-piper-en_GB-alan-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-alan-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-alba-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-jenny_dioco-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-northern_english_male-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-semaine-medium": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-southern_english_female-low": _get_vits_piper,
    "csukuangfj/vits-piper-en_GB-vctk-medium": _get_vits_piper,
}

german_models = {
    "csukuangfj/vits-piper-de_DE-eva_k-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-karlsson-low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-kerstin-low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-pavoque-low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-ramona-low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-thorsten-low": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-thorsten-medium": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-thorsten-high": _get_vits_piper,
    "csukuangfj/vits-piper-de_DE-thorsten_emotional-medium": _get_vits_piper,  # 8 speakers
}


language_to_models = {
    "Chinese": list(chinese_models.keys()),
    "English": list(english_models.keys()),
    "German": list(german_models.keys()),
}
