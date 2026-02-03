#!/usr/bin/env python3

# This file is part of UDPipe 2 <http://github.com/ufal/udpipe>.
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Modified July 24, 2024 Ben Nagy
#
# - wrap client code as a class instead of a commandline tool
# - add some processing to convert most of the text response to a dataframe, which is probably not
#   generally useful, but belongs here in my workflow.

import email.mime.multipart
import email.mime.nonmultipart
import email.policy
import json
import re
import time
import unicodedata
import pandas as pd
import sys
import urllib.error
import urllib.request
from typing import Any
from nltk.tokenize.punkt import PunktSentenceTokenizer
import la_senter
import la_core_web_lg
from spacy.language import Language


def _cap_diff(x: int) -> str:
    if x > 5:
        return "+"
    elif x < -5:
        return "-"
    else:
        return str(x)


def _get_token(x: dict) -> str:
    return f"{x['rel']}_{x['parent_rel']}({x['diff']})"


LT_ABBREV = set(
    [
        "ti",
        "gn",
        "cn",
        "sp",
        "kal",
        "agr",
        "ap",
        "mam",
        "oct",
        "opet",
        "post",
        "pro",
        "ser",
        "st",
        "m",
    ]
)
sent_tok = PunktSentenceTokenizer()
sent_tok._params.abbrev_types = LT_ABBREV
senter = la_senter.load()
senter_fallback = la_senter.load()
tokenizer = la_core_web_lg.load().tokenizer


@Language.component("prevent_colon_split")
def prevent_colon_split(doc):
    for i, token in enumerate(doc[:-1]):
        if token.text == ":":
            # Explicitly prevent the next token from being a sentence start
            doc[i + 1].is_sent_start = False
    return doc


# Add after the senter component
senter.add_pipe("prevent_colon_split", after="senter")


class UDPipeClient:

    def __init__(self, arg_hsh: dict):

        self.args = {
            "input": "conllu",
            "model": "",
            "output": "conllu",
            "parser": "",
            "tagger": "",
            "tokenizer": "",
            "outfile": "",
            "service": "https://lindat.mff.cuni.cz/services/udpipe/api",
        }
        self.args.update(arg_hsh)

    def _perform_request(
        self, method: str, params: dict = {}
    ) -> Any:  # json dot loads LOL LMAO
        if not params:
            request_headers, request_data = {}, None
        else:
            message = email.mime.multipart.MIMEMultipart(
                "form-data", policy=email.policy.HTTP  # type:ignore
            )

            for name, value in params.items():
                payload = email.mime.nonmultipart.MIMENonMultipart("text", "plain")
                payload.add_header(
                    "Content-Disposition", 'form-data; name="{}"'.format(name)
                )
                payload.add_header("Content-Transfer-Encoding", "8bit")
                payload.set_payload(value, charset="utf-8")
                message.attach(payload)

            request_data = message.as_bytes().split(b"\r\n\r\n", maxsplit=1)[1]
            request_headers = {
                "Content-Type": message["Content-Type"],
                "User-Agent": "If I am generating too much traffic, email benjamin.nagy@ijp.pan.pl",
            }

        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    url="{}/{}".format(self.args["service"], method),
                    headers=request_headers,
                    data=request_data,
                )
            ) as request:
                try:
                    resp = request.read().decode("utf-8")
                    return json.loads(resp)
                except Exception as e:
                    print(
                        "Cannot read the response of UDPipe '{}' REST request.\n"
                        "  {}".format(method, repr(e)),
                        file=sys.stderr,
                    )
                    raise
        except urllib.error.HTTPError as e:
            print(
                "An exception was raised during UDPipe 'process' REST request.\n"
                "The service returned the following error:\n"
                "  {}".format(e.fp.read().decode("utf-8")),
                file=sys.stderr,
            )
            raise
        except json.JSONDecodeError as e:
            print(
                "Cannot parse the JSON response of UDPipe 'process' REST request.\n"
                "  {}".format(e.msg),
                file=sys.stderr,
            )
            raise

    def list_models(self):
        """
        Query the service and list all of the available models. Prints the list to stdout.

        Args: None

        Returns: None
        """
        response = self._perform_request("models")
        if "models" in response:
            for model in response["models"]:
                print(model)
        if "default_model" in response:
            print("Default model:", response["default_model"])

    def remove_macrons(self, text: str) -> str:
        """
        Remove macrons from Latin text.

        Args:
            text (`str`): Text to process.
        Returns:
            `str`: Text without macrons.
        """
        return unicodedata.normalize("NFD", text).replace("\u0304", "")

    def process_string(
        self,
        s: str,
        raw: bool = True,
        strip_punct: bool = True,
        presegment: bool = True,
        remove_macrons: bool = True,
    ) -> str | tuple[list[pd.DataFrame], list[str]]:
        """
        Process a raw string and return the service response.

        Args:
            s (`str`): string to process. The service will deal fine with multiline text, mixed case,
            macrons, etc., BUT NOT all punctuation ([]<>...)

            raw (`bool`, default = `True`): whether to return the raw response from the service (the
            same as the Output Text in the UDPipe web interface) or a tuple (list of dataframes,
            list of strings), one per sentence (as determined by the service).

            strip_punct (`bool`, default = `True`): whether to strip the [ ] < > { } † characters from
            the input. These characters are used in the Perseus Digital Library and some other
            corpora to mark uncertain readings, editorial interpolations, etc., but will confuse the
            sentence tokenizer and UDPipe.

            presegment (`bool`, default = `True`): whether to pre-segment the text into sentences using
            nltk's PunktSentenceTokenizer with Latin abbreviations. This is useful if you have a
            large amount of text with few or no newlines, as UDPipe's internal sentence segmentation
            is not great for Latin.

        Returns:
            `str` | `tuple[list[pd.DataFrame], list[str]]`: the result
        """
        if strip_punct:
            s = s.translate(str.maketrans("", "", r"[]<>{}†'\""))
        if remove_macrons:
            s = self.remove_macrons(s)
        data = {
            "input": self.args["input"],
            "output": self.args["output"],
            "data": s,
        }
        for option in ["model", "tokenizer", "parser", "tagger"]:
            value = self.args.get(option)
            if value is not None:
                data[option] = value
        if presegment:
            tok_arg = data["tokenizer"].split(";")
            tok_arg.append("presegmented")
            data["tokenizer"] = ";".join(tok_arg)
            # ary = sent_tok.tokenize(s.replace("\n", " "))
            t = " ".join(s.split())
            ary = [sent.text for sent in senter(t).sents]
            clean = [s for s in ary if re.match(r".*?\w", s)]
            clean = [
                " ".join([token.text for token in list(tokenizer(s))]) for s in clean
            ]
            s = "\n".join(clean)
            data["data"] = s

        try:
            response = self._perform_request("process", data)
        except Exception as e:
            print(f"Error processing request: {e}")
            print("Falling back to alternative sentence tokenizer.")
            ary = [sent.text for sent in senter_fallback(t).sents]
            clean = [s for s in ary if re.match(r".*?\w", s)]
            s = "\n".join(clean)
            data["data"] = s
            # if this fails just let it raise
            response = self._perform_request("process", data)

        if "model" not in response or "result" not in response:
            raise ValueError("Cannot parse the UDPipe 'process' REST request response.")

        if raw:
            return response["result"]
        else:
            return self._process_response(response["result"] + "\n#")

    def _process_response(self, resp: str) -> tuple[list[pd.DataFrame], list[str]]:
        """
        Process the response from the UDPipe server and return a list of dataframes, one per
        sentence.

        Args:
            resp (`str`): The response from the UDPipe server.

        Returns:
            `tuple[list[pd.DataFrame], list[str]]`: A tuple containing a list of dataframes and a list
            of texts.
        """
        # We rely on a terminating '#' character to write out the final frame which is stupid and I
        # am bad at code.

        frames = []
        texts = []
        gather = False
        this_frame = []
        cols = [
            "idx",
            "word",
            "lemma",
            "POS",
            "POS2",
            "Feats",
            "parent",
            "rel",
            "junk",
            "junk2",
        ]
        for l in resp.splitlines():
            try:
                if l and not (l.startswith("#") or re.match(r"^\d+", l)):
                    raise ValueError(f"PANIC: Unknown line type in response: {l}")
                if not l:
                    continue
                if gather:
                    if not re.match(r"^\d+", l):
                        gather = False
                        df = pd.DataFrame(this_frame, columns=cols)
                        # In some cases a word with a clitic gets split into two subsequent tokens and
                        # this 'header word' is inserted, which we don't want.
                        # 49-50	Crannōnisque	_	_	_	_	_	_	_	_
                        # 49	Crannōnis	crannōnus	ADJ	Sms3g	Case=Gen|Gender=Masc...
                        # 50	que	que	CCONJ	9	_	51	cc	_	_
                        df = df[df["parent"] != "_"]
                        df["parent_rel"] = df["parent"].apply(
                            # so the root node will look like root_ISAT(3), etc.
                            lambda x: (
                                "ISAT" if int(x) == 0 else df.iloc[int(x) - 1]["rel"]
                            )
                        )
                        diff = df["idx"].astype(int) - df["parent"].astype(int)
                        df["diff"] = diff.apply(_cap_diff)
                        df["tok_3"] = df.apply(_get_token, axis=1)
                        df.drop(["POS2", "junk", "junk2"], inplace=True, axis=1)
                        stripped_df = df[df["POS"] != "PUNCT"]
                        # Sometimes you have a sentence that is just punctuation :\
                        if len(stripped_df) > 0:
                            frames.append(stripped_df)
                        this_frame.clear()
                    else:
                        this_frame.append(l.split("\t"))
                else:
                    if not l.startswith("# text"):
                        continue
                    else:
                        texts.append(l.split("=")[1].strip())
                        gather = True
            except Exception as e:
                print(l)
                raise e

        return (frames, texts)

    KEEP_FEATURES = [
        "Number",
        "InflClass",
        "Case",
        "Gender",
        "VerbForm",
        "Aspect",
        "Voice",
        "Person",
        "Mood",
        "Tense",
        "PronType",
        "InflClass[nominal]",
    ]

    def process_df(
        self,
        df: pd.DataFrame,
        feats: list[str] = KEEP_FEATURES,
        strip_punct: bool = True,
        presegment: bool = True,
        remove_macrons: bool = True,
    ) -> pd.DataFrame:
        """
        Process a dataframe in the format from `hypotactic.py`, one row per work, and return a
        dataframe one row per sentence with UDPipe information. The UDPipe output will be appended
        in three new columns:
            - `tree` contains a flat representation of the dependeny tree in triples (src_dep,
              dest_dep, distance). E.g., det_obj(1) means a determiner that is one word before an
              object.
            - `feats` contains a list of the UD Features for each word (just concatenated as one
              string). By default only the most common 12 are kept.
            - `pos` contains a string list of each POS in the sentence
        We also append a new column text with the text of the sentence.

        Args:
            df (`pd.DataFrame`): DataFrame to operate on

            feats (`list[str]`), optional: list of UD Feature types (each type can have several
            values) to keep

        Returns:
            (`pd.Dataframe`): New dataframe with the sentence by sentence breakdowns
        """

        # func to apply
        def df_to_trees(row: pd.Series) -> pd.DataFrame:
            # process gives us a list of (dataframe, sentence_text) tuples, one for each sentence in
            # this row (which is a whole text)
            frames, texts = self.process_string(
                row["text"],
                raw=False,
                strip_punct=strip_punct,
                presegment=presegment,
                remove_macrons=remove_macrons,
            )  # type: ignore
            batch = []
            for i, (f, t) in enumerate(zip(frames, texts)):
                # each frame is a sentence, each row is now a word. One word can have a variable number
                # of UDPipe features, separated by | characters.
                assert isinstance(f, pd.DataFrame)
                try:
                    ft = f.Feats.str.split("|").explode()
                    # Keep only the most common/useful ones to avoid getting too sparse
                    keep_feats = ft[ft.str.split("=", expand=True)[0].isin(feats)]
                    keep_feats = [str(x) for x in keep_feats]
                    batch.append(
                        {
                            "title": row["canonical_title"],
                            "author": row["author"],
                            "work": row["work"],
                            "meter": row["meter"],
                            "sentence": f"{row.canonical_title}-{i}",
                            "tree": " ".join(f.tok_3),
                            "feats": " ".join(keep_feats),
                            "pos": " ".join(f.POS),
                            "text": t,
                        }
                    )
                except Exception as e:
                    print(f"Fatal at sentence {i} (Text: {t}): {repr(e)}")
                    raise e
            time.sleep(0.1)  # be polite
            return pd.DataFrame.from_records(batch)

        return pd.concat(list(df.apply(df_to_trees, axis=1))).reset_index(drop=True)  # type: ignore
