"""Prompt construction and host-side embeddings for Qwen3-TTS CustomVoice (speaker + language).

Mirrors `Qwen3TTSForConditionalGeneration.generate` of qwen-tts 0.1.1 with no instruct text and
`non_streaming_mode=False`: the talker's prefill is 10 positions (role tokens, codec think/language/
speaker tags overlaid with tts_pad/tts_bos, the first text token on codec_bos), and the remaining
text tokens plus tts_eos are added one per generated frame ("trailing text"), then tts_pad.

The embedding tables live on the host; sums are done in FP32 and rounded to BF16 once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BF = torch.bfloat16


class Embedder:
    """Tokenizer and host-side embedding tables: builds each request's prefill and trailing text."""

    def __init__(self, cfg: dict, w: dict, root: str):
        from transformers import AutoTokenizer

        self.cfg = cfg
        tc = cfg["talker_config"]
        self.tc = tc
        self.tok = AutoTokenizer.from_pretrained(root)
        self.text_emb = w["talker.model.text_embedding.weight"]  # [151936, 2048] BF16
        self.fc1 = (w["talker.text_projection.linear_fc1.weight"].float(), w["talker.text_projection.linear_fc1.bias"].float())
        self.fc2 = (w["talker.text_projection.linear_fc2.weight"].float(), w["talker.text_projection.linear_fc2.bias"].float())
        self.codec_emb = w["talker.model.codec_embedding.weight"].float()  # [3072, 2048]
        n = tc["code_predictor_config"]["num_code_groups"] - 1
        self.cp_emb = torch.stack([w[f"talker.code_predictor.model.codec_embedding.{i}.weight"] for i in range(n)]).float()  # [15, 2048, 2048]
        special = self.text_proj(torch.tensor([cfg["tts_bos_token_id"], cfg["tts_eos_token_id"], cfg["tts_pad_token_id"]]))
        self.tts_bos, self.tts_eos, self.tts_pad = special[0], special[1], special[2]
        self.eos_id = tc["codec_eos_token_id"]
        self.vocab = tc["vocab_size"]

    def text_proj(self, ids: torch.Tensor) -> torch.Tensor:
        """text_projection(text_embedding(ids)) in FP32: [n, 2048]."""
        e = self.text_emb.index_select(0, ids).float()
        return F.linear(F.silu(F.linear(e, *self.fc1)), *self.fc2)

    def token_ids(self, text: str) -> torch.Tensor:
        s = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        return torch.tensor(self.tok(s)["input_ids"], dtype=torch.long)

    def prompt(self, text: str, speaker: str, language: str):
        """(prefill [10, 2048] BF16, trailing [N, 2048] FP32, number of prompt token ids)."""
        tc = self.tc
        ids = self.token_ids(text)
        spk = speaker.lower()
        if spk not in tc["spk_id"]:
            raise ValueError(f"unknown speaker {speaker}")
        lang = language.lower()
        if lang not in tc["codec_language_id"]:
            raise ValueError(f"unknown language {language}")
        lang_id = tc["codec_language_id"][lang]
        if lang in ("chinese", "auto") and tc["spk_is_dialect"][spk] is not False:
            lang_id = tc["codec_language_id"][tc["spk_is_dialect"][spk]]
        codec = [tc["codec_think_id"], tc["codec_think_bos_id"], lang_id, tc["codec_think_eos_id"], tc["spk_id"][spk], tc["codec_pad_id"], tc["codec_bos_id"]]
        cemb = self.codec_emb[torch.tensor(codec)]  # [7, 2048]
        role = self.text_proj(ids[:3])  # <|im_start|>assistant\n
        overlay = torch.cat([self.tts_pad[None].expand(len(codec) - 2, -1), self.tts_bos[None]]) + cemb[:-1]
        first = self.text_proj(ids[3:4]) + cemb[-1:]
        prefill = torch.cat([role, overlay, first]).to(BF)
        trailing = torch.cat([self.text_proj(ids[4:-5]), self.tts_eos[None]])
        return prefill, trailing, int(ids.shape[0])

    def frame_input(self, codes: list[int], trailing: torch.Tensor, step: int) -> torch.Tensor:
        """The talker input after a frame: the sum of its 16 code embeddings plus the text slot."""
        e = self.codec_emb[codes[0]].clone()
        for g in range(1, len(codes)):
            e += self.cp_emb[g - 1, codes[g]]
        e += trailing[step] if step < trailing.shape[0] else self.tts_pad
        return e.to(BF)

    def code0_embed(self, code: int) -> torch.Tensor:
        return self.codec_emb[code].to(BF)

    def cp_embed(self, g: int, code: int) -> torch.Tensor:
        """Embedding of code g (1..14) as code predictor input (talker width, before projection)."""
        return self.cp_emb[g - 1, code].to(BF)
