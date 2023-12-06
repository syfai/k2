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
    data_dir = "/tmp/espeak-ng-data"
    if "coqui" in repo_id:
        name = "model"
    else:
        n = len("vits-piper-")
        name = repo_id.split("/")[1][n:]

    if "vits-coqui-uk-mai" in repo_id:
        data_dir = ""

    model = get_file(
        repo_id=repo_id,
        filename=f"{name}.onnx",
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
                lexicon="",
                data_dir=data_dir,
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

    rule_fst = get_file(
        repo_id=repo_id,
        filename="rule.fst",
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
        ),
        rule_fsts=rule_fst,
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def _get_vits_hf(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    if "fanchen" in repo_id or "vits-cantonese-hf-xiaomaiiwn" in repo_id:
        model = repo_id.split("/")[-1]
    else:
        model = repo_id.split("-")[-1]

    model = get_file(
        repo_id=repo_id,
        filename=f"{model}.onnx",
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

    rule_fst = get_file(
        repo_id=repo_id,
        filename="rule.fst",
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
        ),
        rule_fsts=rule_fst,
    )
    tts = sherpa_onnx.OfflineTts(tts_config)

    return tts


@lru_cache(maxsize=10)
def get_pretrained_model(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    if repo_id in chinese_models:
        return chinese_models[repo_id](repo_id, speed)
    if repo_id in cantonese_models:
        return cantonese_models[repo_id](repo_id, speed)
    elif repo_id in english_models:
        return english_models[repo_id](repo_id, speed)
    elif repo_id in german_models:
        return german_models[repo_id](repo_id, speed)
    elif repo_id in spanish_models:
        return spanish_models[repo_id](repo_id, speed)
    elif repo_id in french_models:
        return french_models[repo_id](repo_id, speed)
    elif repo_id in ukrainian_models:
        return ukrainian_models[repo_id](repo_id, speed)
    elif repo_id in russian_models:
        return russian_models[repo_id](repo_id, speed)
    elif repo_id in arabic_models:
        return arabic_models[repo_id](repo_id, speed)
    elif repo_id in catalan_models:
        return catalan_models[repo_id](repo_id, speed)
    elif repo_id in czech_models:
        return czech_models[repo_id](repo_id, speed)
    elif repo_id in danish_models:
        return danish_models[repo_id](repo_id, speed)
    elif repo_id in greek_models:
        return greek_models[repo_id](repo_id, speed)
    elif repo_id in finnish_models:
        return finnish_models[repo_id](repo_id, speed)
    elif repo_id in hungarian_models:
        return hungarian_models[repo_id](repo_id, speed)
    elif repo_id in icelandic_models:
        return icelandic_models[repo_id](repo_id, speed)
    elif repo_id in italian_models:
        return italian_models[repo_id](repo_id, speed)
    elif repo_id in georgian_models:
        return georgian_models[repo_id](repo_id, speed)
    elif repo_id in kazakh_models:
        return kazakh_models[repo_id](repo_id, speed)
    elif repo_id in luxembourgish_models:
        return luxembourgish_models[repo_id](repo_id, speed)
    elif repo_id in nepali_models:
        return nepali_models[repo_id](repo_id, speed)
    elif repo_id in dutch_models:
        return dutch_models[repo_id](repo_id, speed)
    elif repo_id in norwegian_models:
        return norwegian_models[repo_id](repo_id, speed)
    elif repo_id in polish_models:
        return polish_models[repo_id](repo_id, speed)
    elif repo_id in portuguese_models:
        return portuguese_models[repo_id](repo_id, speed)
    elif repo_id in romanian_models:
        return romanian_models[repo_id](repo_id, speed)
    elif repo_id in slovak_models:
        return slovak_models[repo_id](repo_id, speed)
    elif repo_id in serbian_models:
        return serbian_models[repo_id](repo_id, speed)
    elif repo_id in swedish_models:
        return swedish_models[repo_id](repo_id, speed)
    elif repo_id in swahili_models:
        return swahili_models[repo_id](repo_id, speed)
    elif repo_id in turkish_models:
        return turkish_models[repo_id](repo_id, speed)
    elif repo_id in vietnamese_models:
        return vietnamese_models[repo_id](repo_id, speed)
    elif repo_id in bulgarian_models:
        return bulgarian_models[repo_id](repo_id, speed)
    elif repo_id in estonian_models:
        return estonian_models[repo_id](repo_id, speed)
    elif repo_id in irish_models:
        return irish_models[repo_id](repo_id, speed)
    elif repo_id in croatian_models:
        return croatian_models[repo_id](repo_id, speed)
    elif repo_id in lithuanian_models:
        return lithuanian_models[repo_id](repo_id, speed)
    elif repo_id in latvian_models:
        return latvian_models[repo_id](repo_id, speed)
    elif repo_id in maltese_models:
        return maltese_models[repo_id](repo_id, speed)
    elif repo_id in slovenian_models:
        return slovenian_models[repo_id](repo_id, speed)
    elif repo_id in bangla_models:
        return bangla_models[repo_id](repo_id, speed)
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


cantonese_models = {
    "csukuangfj/vits-cantonese-hf-xiaomaiiwn": _get_vits_hf,
}

chinese_models = {
    "csukuangfj/vits-piper-zh_CN-huayan-medium": _get_vits_piper,
    "csukuangfj/vits-zh-hf-theresa": _get_vits_hf,
    "csukuangfj/vits-zh-hf-eula": _get_vits_hf,
    "csukuangfj/vits-zh-hf-echo": _get_vits_hf,
    "csukuangfj/vits-zh-hf-bronya": _get_vits_hf,
    "csukuangfj/vits-zh-aishell3": _get_vits_zh_aishell3,
    "csukuangfj/vits-zh-hf-fanchen-wnj": _get_vits_hf,
    "csukuangfj/vits-zh-hf-fanchen-C": _get_vits_hf,
    "csukuangfj/vits-zh-hf-fanchen-ZhiHuiLaoZhe": _get_vits_hf,
    "csukuangfj/vits-zh-hf-fanchen-ZhiHuiLaoZhe_new": _get_vits_hf,
    "csukuangfj/vits-zh-hf-fanchen-unity": _get_vits_hf,
    "csukuangfj/vits-zh-hf-doom": _get_vits_hf,
    "csukuangfj/vits-zh-hf-zenyatta": _get_vits_hf,  # 804
    "csukuangfj/vits-zh-hf-abyssinvoker": _get_vits_hf,
    "csukuangfj/vits-zh-hf-keqing": _get_vits_hf,
    #  "csukuangfj/vits-piper-zh_CN-huayan-x_low": _get_vits_piper,
}

english_models = {
    # coqui-ai
    "csukuangfj/vits-coqui-en-ljspeech": _get_vits_piper,
    "csukuangfj/vits-coqui-en-ljspeech-neon": _get_vits_piper,
    "csukuangfj/vits-coqui-en-vctk": _get_vits_piper,
    # piper, US
    "csukuangfj/vits-piper-en_GB-sweetbbak-amy": _get_vits_piper,
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
    #
    "csukuangfj/vits-vctk": _get_vits_vctk,  # 109 speakers
    "csukuangfj/vits-ljs": _get_vits_ljs,
}

german_models = {
    "csukuangfj/vits-coqui-de-css10": _get_vits_piper,
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

spanish_models = {
    "csukuangfj/vits-coqui-es-css10": _get_vits_piper,
    "csukuangfj/vits-piper-es_ES-carlfm-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-es_ES-davefx-medium": _get_vits_piper,
    "csukuangfj/vits-piper-es_ES-mls_10246-low": _get_vits_piper,
    "csukuangfj/vits-piper-es_ES-mls_9972-low": _get_vits_piper,
    "csukuangfj/vits-piper-es_ES-sharvard-medium": _get_vits_piper,  # 2 speakers
    "csukuangfj/vits-piper-es_MX-ald-medium": _get_vits_piper,
}

french_models = {
    "csukuangfj/vits-coqui-fr-css10": _get_vits_piper,
    #  "csukuangfj/vits-piper-fr_FR-gilles-low": _get_vits_piper,
    #  "csukuangfj/vits-piper-fr_FR-mls_1840-low": _get_vits_piper,
    "csukuangfj/vits-piper-fr_FR-upmc-medium": _get_vits_piper,  # 2 speakers, 0-femal, 1-male
    "csukuangfj/vits-piper-fr_FR-siwis-low": _get_vits_piper,  # female
    "csukuangfj/vits-piper-fr_FR-siwis-medium": _get_vits_piper,
    "csukuangfj/vits-piper-fr_FR-tjiho-model1": _get_vits_piper,
    "csukuangfj/vits-piper-fr_FR-tjiho-model2": _get_vits_piper,
    "csukuangfj/vits-piper-fr_FR-tjiho-model3": _get_vits_piper,
}

ukrainian_models = {
    "csukuangfj/vits-piper-uk_UA-lada-x_low": _get_vits_piper,
    "csukuangfj/vits-coqui-uk-mai": _get_vits_piper,
    #  "csukuangfj/vits-piper-uk_UA-ukrainian_tts-medium": _get_vits_piper, # does not work somehow
}

russian_models = {
    "csukuangfj/vits-piper-ru_RU-denis-medium": _get_vits_piper,
    "csukuangfj/vits-piper-ru_RU-dmitri-medium": _get_vits_piper,
    "csukuangfj/vits-piper-ru_RU-irina-medium": _get_vits_piper,
    "csukuangfj/vits-piper-ru_RU-ruslan-medium": _get_vits_piper,
}

arabic_models = {
    "csukuangfj/vits-piper-ar_JO-kareem-low": _get_vits_piper,
    "csukuangfj/vits-piper-ar_JO-kareem-medium": _get_vits_piper,
}

catalan_models = {
    "csukuangfj/vits-piper-ca_ES-upc_ona-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-ca_ES-upc_ona-medium": _get_vits_piper,
    "csukuangfj/vits-piper-ca_ES-upc_pau-x_low": _get_vits_piper,
}

czech_models = {
    "csukuangfj/vits-piper-cs_CZ-jirka-low": _get_vits_piper,
    "csukuangfj/vits-piper-cs_CZ-jirka-medium": _get_vits_piper,
    "csukuangfj/vits-coqui-cs-cv": _get_vits_piper,
}

danish_models = {
    "csukuangfj/vits-coqui-da-cv": _get_vits_piper,
    "csukuangfj/vits-piper-da_DK-talesyntese-medium": _get_vits_piper,
}

greek_models = {
    "csukuangfj/vits-piper-el_GR-rapunzelina-low": _get_vits_piper,
}

finnish_models = {
    "csukuangfj/vits-coqui-fi-css10": _get_vits_piper,
    "csukuangfj/vits-piper-fi_FI-harri-low": _get_vits_piper,
    "csukuangfj/vits-piper-fi_FI-harri-medium": _get_vits_piper,
}

hungarian_models = {
    #  "csukuangfj/vits-coqui-hu-css10": _get_vits_piper,
    "csukuangfj/vits-piper-hu_HU-anna-medium": _get_vits_piper,
    "csukuangfj/vits-piper-hu_HU-berta-medium": _get_vits_piper,
    "csukuangfj/vits-piper-hu_HU-imre-medium": _get_vits_piper,
}

icelandic_models = {
    "csukuangfj/vits-piper-is_IS-bui-medium": _get_vits_piper,
    "csukuangfj/vits-piper-is_IS-salka-medium": _get_vits_piper,
    "csukuangfj/vits-piper-is_IS-steinn-medium": _get_vits_piper,
    "csukuangfj/vits-piper-is_IS-ugla-medium": _get_vits_piper,
}

italian_models = {
    "csukuangfj/vits-piper-it_IT-riccardo-x_low": _get_vits_piper,
}

georgian_models = {
    "csukuangfj/vits-piper-ka_GE-natia-medium": _get_vits_piper,
}

kazakh_models = {
    "csukuangfj/vits-piper-kk_KZ-iseke-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-kk_KZ-issai-high": _get_vits_piper,
    "csukuangfj/vits-piper-kk_KZ-raya-x_low": _get_vits_piper,
}

luxembourgish_models = {
    "csukuangfj/vits-piper-lb_LU-marylux-medium": _get_vits_piper,
}

nepali_models = {
    "csukuangfj/vits-piper-ne_NP-google-medium": _get_vits_piper,
    "csukuangfj/vits-piper-ne_NP-google-x_low": _get_vits_piper,
}

dutch_models = {
    "csukuangfj/vits-coqui-nl-css10": _get_vits_piper,
    "csukuangfj/vits-piper-nl_BE-nathalie-medium": _get_vits_piper,
    "csukuangfj/vits-piper-nl_BE-nathalie-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-nl_BE-rdh-medium": _get_vits_piper,
    "csukuangfj/vits-piper-nl_BE-rdh-x_low": _get_vits_piper,
    "csukuangfj/vits-piper-nl_NL-mls_5809-low": _get_vits_piper,
    "csukuangfj/vits-piper-nl_NL-mls_7432-low": _get_vits_piper,
}

norwegian_models = {
    "csukuangfj/vits-piper-no_NO-talesyntese-medium": _get_vits_piper,
}

polish_models = {
    "csukuangfj/vits-coqui-pl-mai_female": _get_vits_piper,
    "csukuangfj/vits-piper-pl_PL-darkman-medium": _get_vits_piper,
    "csukuangfj/vits-piper-pl_PL-gosia-medium": _get_vits_piper,
    "csukuangfj/vits-piper-pl_PL-mc_speech-medium": _get_vits_piper,
    "csukuangfj/vits-piper-pl_PL-mls_6892-low": _get_vits_piper,
}

portuguese_models = {
    "csukuangfj/vits-coqui-pt-cv": _get_vits_piper,
    "csukuangfj/vits-piper-pt_BR-edresson-low": _get_vits_piper,
    "csukuangfj/vits-piper-pt_BR-faber-medium": _get_vits_piper,
    "csukuangfj/vits-piper-pt_PT-tugao-medium": _get_vits_piper,
}

romanian_models = {
    "csukuangfj/vits-coqui-ro-cv": _get_vits_piper,
    "csukuangfj/vits-piper-ro_RO-mihai-medium": _get_vits_piper,
}


slovak_models = {
    "csukuangfj/vits-coqui-sk-cv": _get_vits_piper,
    "csukuangfj/vits-piper-sk_SK-lili-medium": _get_vits_piper,
}

serbian_models = {
    "csukuangfj/vits-piper-sr_RS-serbski_institut-medium": _get_vits_piper,
}

swedish_models = {
    "csukuangfj/vits-coqui-sv-cv": _get_vits_piper,
    "csukuangfj/vits-piper-sv_SE-nst-medium": _get_vits_piper,
}

swahili_models = {
    "csukuangfj/vits-piper-sw_CD-lanfrica-medium": _get_vits_piper,
}

turkish_models = {
    "csukuangfj/vits-piper-tr_TR-dfki-medium": _get_vits_piper,
    "csukuangfj/vits-piper-tr_TR-fahrettin-medium": _get_vits_piper,
}

vietnamese_models = {
    "csukuangfj/vits-piper-vi_VN-25hours_single-low": _get_vits_piper,
    "csukuangfj/vits-piper-vi_VN-vais1000-medium": _get_vits_piper,
    "csukuangfj/vits-piper-vi_VN-vivos-x_low": _get_vits_piper,
}

bulgarian_models = {
    "csukuangfj/vits-coqui-bg-cv": _get_vits_piper,
}

estonian_models = {
    "csukuangfj/vits-coqui-et-cv": _get_vits_piper,
}

irish_models = {
    "csukuangfj/vits-coqui-ga-cv": _get_vits_piper,
}

croatian_models = {
    "csukuangfj/vits-coqui-hr-cv": _get_vits_piper,
}

lithuanian_models = {
    "csukuangfj/vits-coqui-lt-cv": _get_vits_piper,
}

latvian_models = {
    "csukuangfj/vits-coqui-lv-cv": _get_vits_piper,
}

maltese_models = {
    "csukuangfj/vits-coqui-mt-cv": _get_vits_piper,
}

slovenian_models = {
    "csukuangfj/vits-coqui-sl-cv": _get_vits_piper,
}

bangla_models = {
    "csukuangfj/vits-coqui-bn-custom_female": _get_vits_piper,
}


language_to_models = {
    "English": list(english_models.keys()),
    "Chinese (Mandarin, 普通话)": list(chinese_models.keys()),
    "Cantonese (粤语)": list(cantonese_models.keys()),
    "Arabic": list(arabic_models.keys()),
    "Bangla": list(bangla_models.keys()),
    "Bulgarian": list(bulgarian_models.keys()),
    "Catalan": list(catalan_models.keys()),
    "Croatian": list(croatian_models.keys()),
    "Czech": list(czech_models.keys()),
    "Danish": list(danish_models.keys()),
    "Dutch": list(dutch_models.keys()),
    "Estonian": list(estonian_models.keys()),
    "Finnish": list(finnish_models.keys()),
    "French": list(french_models.keys()),
    "Georgian": list(georgian_models.keys()),
    "German": list(german_models.keys()),
    "Greek": list(greek_models.keys()),
    "Hungarian": list(hungarian_models.keys()),
    "Icelandic": list(icelandic_models.keys()),
    "Irish": list(irish_models.keys()),
    "Italian": list(italian_models.keys()),
    "Kazakh": list(kazakh_models.keys()),
    "Latvian": list(latvian_models.keys()),
    "Lithuanian": list(lithuanian_models.keys()),
    "Luxembourgish": list(luxembourgish_models.keys()),
    "Maltese": list(maltese_models.keys()),
    "Nepali": list(nepali_models.keys()),
    "Norwegian": list(norwegian_models.keys()),
    "Polish": list(polish_models.keys()),
    "Portuguese": list(portuguese_models.keys()),
    "Romanian": list(romanian_models.keys()),
    "Russian": list(russian_models.keys()),
    "Serbian": list(serbian_models.keys()),
    "Slovak": list(slovak_models.keys()),
    "Slovenian": list(slovenian_models.keys()),
    "Spanish": list(spanish_models.keys()),
    "Swahili": list(swahili_models.keys()),
    "Swedish": list(swedish_models.keys()),
    "Turkish": list(turkish_models.keys()),
    "Ukrainian": list(ukrainian_models.keys()),
    "Vietnamese": list(vietnamese_models.keys()),
}
