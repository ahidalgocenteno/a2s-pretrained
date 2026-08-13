import re
from typing import List, Union

import numpy as np

VOICE_CHANGE_TOKEN = "<COC>"
STEP_CHANGE_TOKEN = "<COR>"


class krnParser:
    """Main Kern parser operations class."""

    def __init__(self, use_voice_change_token: bool = True) -> None:
        self.reserved_words = ["clef", "k[", "*M"]
        self.reserved_dot = "."
        self.reserved_dot_EncodedCharacter = "DOT"
        self.clef_change_other_voices = "*"
        self.comment_symbols = ["*", "!"]
        self.voice_change = VOICE_CHANGE_TOKEN  # change-of-column (coc) token
        self.step_change = STEP_CHANGE_TOKEN  # change-of-row (cor) token
        self.use_voice_change_token = use_voice_change_token

    # ---------------------------------------------------------------------------- AUXILIARY FUNCTIONS

    def _readSrcFile(self, file_path: str | None = None, text: str | None = None) -> np.ndarray:
        """Read polyphonic kern file and adequate the format for further processes."""
        if text is not None:
            in_src = text.splitlines()
        elif file_path is not None:
            with open(file_path) as fin:
                in_src = fin.read().splitlines()
        else:
            raise ValueError("Either file_path or text must be provided.")

        # Locating line with the headers
        it_headers = 0
        while "**kern" not in in_src[it_headers]:
            it_headers += 1
        pass
        columns_to_process = np.where(
            np.array(in_src[it_headers].split("\t")) == "**kern"
        )[0]

        # Locating lines with comments (to be removed)
        in_src_nocomments = []
        for line in in_src:
            if not line.strip().startswith("!"):
                in_src_nocomments.append(line.split("\t"))
        pass

        # Extract voices and removing lines with comments
        out_src = np.array(in_src_nocomments)[:, columns_to_process]

        return out_src

    def _postprocessKernSequence(self, in_score: np.ndarray) -> np.ndarray:
        """Exchanging '*' for the actual symbol."""

        # Retrieving positions with '*'
        positions = sorted(list(set(np.where(in_score == "*")[1])))

        # For each position,
        # we retrieve the last explicit clef symbol and include it in the stream
        for single_position in positions:
            for it_voice in range(in_score.shape[0]):
                if in_score[it_voice, single_position] == "*":
                    new_element = in_score[
                        it_voice,
                        max(
                            np.where(np.char.startswith(in_score[it_voice], "*clef"))[0]
                        ),
                    ]
                    in_score[it_voice, single_position] = new_element
                pass
            pass
        pass

        return in_score

    def cleanKernFile(self, file_path: str | None = None, text: str | None = None) -> np.ndarray:
        """Convert complete kern sequence to CLEAN kern format."""
        in_file = self._readSrcFile(file_path=file_path, text=text)

        # Processing individual voices
        out_score = []
        for it_voice in range(in_file.shape[1]):
            in_voice = in_file[:, it_voice].tolist()
            out_voice = [
                self.cleanKernToken(u)
                for u in in_voice
                if self.cleanKernToken(u) is not None
            ]

            out_score.append(out_voice)
        pass
        out_score = np.array(out_score)

        # Postprocess obtained score
        out_score = self._postprocessKernSequence(out_score)

        return out_score

    def cleanKernToken(self, in_token: str) -> Union[str, None]:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default

        if in_token != "":
            if any(
                [u in in_token for u in self.reserved_words]
            ):  # Relevant reserved tokens
                out_token = in_token

            elif in_token == self.reserved_dot:  # Case when using "." for sync. voices
                out_token = self.reserved_dot_EncodedCharacter

            elif (
                in_token.strip() == self.clef_change_other_voices
            ):  # Clef change in other voices
                out_token = in_token

            elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
                out_token = None

            elif in_token.startswith("s"):  # Slurs
                out_token = "s"

            elif "=" in in_token:  # Bar lines
                out_token = "="

            elif not "q" in in_token:
                if "rr" in in_token:  # Multirest
                    out_token = re.findall(r"rr[0-9]+", in_token)[0]
                elif "r" in in_token:  # Rest
                    out_token = in_token.split("r")[0] + "r"
                else:  # Music note
                    out_token = re.findall(r"\[*\d+[.]*[a-gA-G]+[n#-]*\]*", in_token)[0]

        return out_token

    # ---------------------------------------------------------------------------- CONVERT CALL

    def convert(self, src_file: str | None = None , text: str | None = None) -> List[str]:
        assert src_file is not None or text is not None, "Either src_file or text must be provided."
        
        if text is not None:
            # Process the provided text directly
            out = self.cleanKernFile(text=text).T
        else:
            # Process the file
            out = self.cleanKernFile(file_path=src_file).T

        out_line = []
        for t in out:
            for v in t:
                out_line.append(v)
                if self.use_voice_change_token:
                    out_line.append(self.voice_change)
            if self.use_voice_change_token:
                del out_line[-1]
            out_line.append(self.step_change)
        del out_line[-1]

        return out_line

    def detokenize_to_string(self,
                             token_sequence: Union[List[str], np.ndarray, str],
                             n_voices: int = None,
                             include_header: bool = True,
                             write_path: str = None,
                             pad_with: str = None) -> str:
        """
        Reconstruct a kern-like string from a token stream produced by convert().

        Returns a string where:
          - each step (row) is separated by a newline '\n'
          - each voice (column) within a step is separated by a tab '\t'

        Args:
            token_sequence: list/1D-array/space-separated string of tokens (including <COC> and <COR> if used).
            n_voices: optional, number of voices to assume when voice-change token is NOT used.
                      If None, will be inferred from longest row.
            include_header: if True, include a header line with "**kern" repeated for each voice.
            write_path: optional path to write the resulting string to a file.
            pad_with: optional token to use to fill missing voice cells (default uses self.reserved_dot).

        Returns:
            A string ready to be written as a .krn-like tab-separated file.
        """
        # Normalize input tokens into a list
        if isinstance(token_sequence, str):
            tokens = token_sequence.split()
        elif isinstance(token_sequence, np.ndarray):
            tokens = token_sequence.tolist()
        else:
            tokens = list(token_sequence)

        if pad_with is None:
            pad_with = self.reserved_dot  # default '.' continuation

        # if the last token is <EOS>, remove it
        if len(tokens) > 0 and tokens[-1] == "<EOS>":
            tokens = tokens[:-1]

        # Split into raw rows by STEP_CHANGE_TOKEN
        rows_raw = []
        cur = []
        for tok in tokens:
            if tok == self.step_change:
                rows_raw.append(cur)
                cur = []
            else:
                cur.append(tok)
        # append last row if there was no trailing <COR>
        if cur:
            rows_raw.append(cur)

        # Parse each row into a list of voice-cells
        rows_parsed = []
        for row in rows_raw:
            if self.use_voice_change_token:
                voices = []
                cur_voice = []
                for tok in row:
                    if tok == self.voice_change:
                        voices.append(cur_voice)
                        cur_voice = []
                    else:
                        cur_voice.append(tok)
                voices.append(cur_voice)
                # collapse tokens inside a voice cell into a single string:
                # - if empty -> pad_with
                # - if one token -> that token
                # - if multiple -> join without separator (mirrors original linking)
                voices = [
                    (pad_with if len(v) == 0 else (v[0] if len(v) == 1 else "".join(v)))
                    for v in voices
                ]
            else:
                # No voice-change token present: treat the row tokens as already being voice cells
                voices = list(row) if len(row) > 0 else [pad_with]
            rows_parsed.append(voices)

        # Decide number of voices (columns)
        if self.use_voice_change_token:
            max_voices = max((len(r) for r in rows_parsed), default=0)
        else:
            if n_voices is None:
                max_voices = max((len(r) for r in rows_parsed), default=0)
                n_voices = max_voices
            else:
                max_voices = n_voices

        # Normalize/pad rows to have same number of voices and convert encoded DOT back to '.'
        normalized_rows = []
        for r in rows_parsed:
            if len(r) < max_voices:
                r = r + [pad_with] * (max_voices - len(r))
            # Replace encoded DOT token back to actual dot for continuity
            r = [
                (self.reserved_dot if (cell == self.reserved_dot_EncodedCharacter) else cell)
                for cell in r
            ]
            normalized_rows.append(r)

        # Build lines (each line = one step). If there are no rows, return empty string or header only.
        lines = []
        if include_header and max_voices > 0:
            header = "\t".join(["**kern"] * max_voices)
            lines.append(header)

        for step_row in normalized_rows:
            # ensure all cells are str
            step_row_str = [("" if cell is None else str(cell)) for cell in step_row]
            lines.append("\t".join(step_row_str))

        result = "\n".join(lines)

        # Optionally write to file
        if write_path:
            with open(write_path, "w", encoding="utf-8") as fout:
                fout.write(result)

        return result


class semanticAgnosticParser:
    def convert(self, file_path: str) -> List[str]:
        with open(file_path) as fin:
            in_src = fin.read()
        in_src = in_src.replace("\t", " ")
        in_src = in_src.split()
        return in_src