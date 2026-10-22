"""`AutoModel.from_pretrained(repo, trust_remote_code=True)`: d1-omni with the System One API.

    model.system_one(state, {name: question}, images=None, audio=None)
    model.system_one_batch([(state, {name: question}[, images[, audio]]), ...])

A state is a string or any JSON value. `images` is a PIL image or a list of them (in order); `audio` is one 16 kHz mono
clip (int16 PCM or float samples). A request carries images or audio, not both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .audio import Audio
from .encoder import DecisionHead, Trunk
from .prompt import QTYPES, Question, answer, as_question, encode, temperature_key
from .vision import Vision

YES_NO = {"false": "no", "true": "yes"}  # how image and audio questions were trained


class D1OmniConfig(PretrainedConfig):
    model_type = "d1_omni"

    def __init__(self, text_config=None, vision_config=None, audio_config=None, projector_hidden_size=2048,
                 head_layers=2, max_length=16384, image_text_length=896,
                 audio_text_length=15360, temperatures=None, **kwargs):
        self.text_config, self.vision_config, self.audio_config = text_config or {}, vision_config or {}, audio_config or {}
        self.projector_hidden_size = projector_hidden_size
        self.head_layers, self.max_length = head_layers, max_length
        self.image_text_length, self.audio_text_length = image_text_length, audio_text_length
        self.temperatures = temperatures or {}
        super().__init__(**kwargs)


class D1OmniModel(PreTrainedModel):
    config_class = D1OmniConfig
    base_model_prefix = "d1"
    main_input_name = "input_ids"

    def __init__(self, config: D1OmniConfig):
        super().__init__(config)
        d = config.text_config["hidden_size"]
        self.encoder = Trunk(config.text_config)
        self.head = DecisionHead(d, config.head_layers)
        self.vision = Vision(config.vision_config, config.projector_hidden_size, d)
        self.audio = Audio(config.audio_config, d)
        self.post_init()

    def _init_weights(self, module):
        pass

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)

    # ------------------------------------------------------------------ API

    def system_one(self, state: Any, questions: Mapping[str, Any], images: Sequence | None = None,
                   audio=None) -> dict:
        """Named questions over one state, and its images or audio if any:
        `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`."""
        return self.system_one_batch([(state, questions, images, audio)])[0]

    def system_one_batch(self, requests: Sequence[tuple]) -> list[dict]:
        """`(state, questions)`, `(state, questions, images)` or `(state, questions, images, audio)` requests."""
        named = [{n: as_question(q) for n, q in r[1].items()} for r in requests]
        done = self.probabilities_batch([(r[0], list(qs.values()), *r[2:]) for r, qs in zip(requests, named)],
                                        _usage=True)
        return [{"answers": {n: answer(q, p) for (n, q), p in zip(qs.items(), probs)},
                 "usage": {"input_tokens": read, "output_tokens": 0}}
                for qs, (probs, read) in zip(named, done)]

    def probabilities(self, state: Any, questions: Sequence, images: Sequence | None = None, audio=None) -> list:
        """Each question's distribution over its options, in option order (`yes`, `no` for a noul)."""
        return self.probabilities_batch([(state, questions, images, audio)])[0]

    @torch.no_grad()
    def probabilities_batch(self, requests: Sequence[tuple], _usage: bool = False) -> list:
        tok, rows, read = self.tokenizer, [], []
        for r in requests:
            state, questions = r[0], [as_question(q) for q in r[1]]
            images, audio = (r[2] if len(r) > 2 else None) or None, r[3] if len(r) > 3 else None
            if images is not None and audio is not None:
                raise ValueError("a request carries images or audio, not both")
            if images is not None:
                prefix = self.vision([images] if hasattr(images, "convert") else list(images))
                max_len, noul, calibrate, spoken = self.config.image_text_length, YES_NO, False, False
            elif audio is not None:
                prefix = self.audio(audio)
                max_len, noul, calibrate, spoken = self.config.audio_text_length, YES_NO, False, True
                state = {} if state is None else state  # how the audio questions were trained
            else:
                prefix, max_len, noul, calibrate, spoken = None, self.config.max_length, None, True, False
            p = 0 if prefix is None else prefix.shape[1]
            max_len = min(max_len, self.config.max_length - p)
            if max_len < 64:
                raise ValueError(f"the media take {p} of the {self.config.max_length} positions; send fewer images")
            state = "" if state is None else state
            seqs = [encode(tok, state, q, max_len, noul, spoken) for q in questions]
            rows.append([(prefix, ids, markers, q, calibrate) for q, (ids, markers) in zip(questions, seqs)])
            read.append(sum(p + len(ids) for ids, _ in seqs))
        flat = [row for request in rows for row in request]
        probs = self._run(flat)
        out, i = [], 0
        for request in rows:
            out.append(probs[i:i + len(request)])
            i += len(request)
        return list(zip(out, read)) if _usage else out

    # ------------------------------------------------------------------ forward

    def _run(self, rows: list, max_tokens: int = 65536) -> list:
        """Rows (prefix, ids, markers, question, calibrate) -> probabilities, batched by token budget."""
        results = [None] * len(rows)
        order = sorted(range(len(rows)), key=lambda i: _length(rows[i]), reverse=True)
        while order:
            longest, size = _length(rows[order[0]]), 1
            while size < len(order) and (size + 1) * longest <= max_tokens:
                size += 1
            batch, order = order[:size], order[size:]
            for i, p in zip(batch, self._forward([rows[i] for i in batch])):
                results[i] = p
        return results

    def _forward(self, rows: list) -> list:
        device, dtype = self.device, self.encoder.embed_tokens.weight.dtype
        seqs, lengths, offsets = [], [], []
        for prefix, ids, *_ in rows:
            text = self.encoder.embed_tokens(torch.tensor(ids, device=device))
            seqs.append(text if prefix is None else torch.cat([prefix[0].to(dtype), text]))
            offsets.append(0 if prefix is None else prefix.shape[1])
            lengths.append(len(ids))
        h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        total = torch.tensor([len(s) for s in seqs], device=device)
        pad = torch.arange(h.shape[1], device=device)[None] < total[:, None]
        h = self.encoder(h, pad, torch.tensor(offsets, device=device))
        text = torch.nn.utils.rnn.pad_sequence([h[i, o:o + n] for i, (o, n) in enumerate(zip(offsets, lengths))],
                                               batch_first=True)
        k = max(len(r[2]) for r in rows)
        text_pad = torch.arange(text.shape[1], device=device)[None] < torch.tensor(lengths, device=device)[:, None]
        marker_pos = torch.zeros(len(rows), k, dtype=torch.long, device=device)
        marker_mask = torch.zeros(len(rows), k, dtype=torch.bool, device=device)
        for i, r in enumerate(rows):
            marker_pos[i, :len(r[2])] = torch.tensor(r[2], device=device)
            marker_mask[i, :len(r[2])] = True
        qtype = torch.tensor([QTYPES[r[3].type] for r in rows], device=device)
        logits = self.head(text, text_pad, marker_pos, marker_mask, qtype)
        out = []
        for row, z in zip(rows, logits):
            q: Question = row[3]
            z = z[:q.options]
            if row[4]:
                z = z / self.config.temperatures.get(temperature_key(q), self.config.temperatures.get(q.type, 1.0))
            p = z.softmax(-1).tolist()
            out.append(p[::-1] if q.type == "noul" else p)  # the model reads a noul as [false, true]
        return out


def _length(row) -> int:
    return (0 if row[0] is None else row[0].shape[1]) + len(row[1])
