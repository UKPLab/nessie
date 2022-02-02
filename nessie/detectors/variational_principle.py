import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterator, List, Tuple

import awkward as ak
import numpy as np
import numpy.typing as npt
from iobes import SpanEncoding, parse_spans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from suffix_tree import Tree
from tqdm import tqdm

from nessie.detectors.error_detector import Detector, DetectorKind
from nessie.types import RaggedStringArray


class VariationNGrams(Detector):
    """Detecting Inconsistencies in Treebanks
    Markus Dickinson and Walt Detmar Meurers
    Proceedings of the Second Workshop on Treebanks and Linguistic Theories (TLT 2003). Växjö, Sweden.
    """

    def score(self, sentences: RaggedStringArray, tags: RaggedStringArray, **kwargs) -> ak.Array:
        """Collects n-grams and their respective label sequences, if there are disagreements, then flag
        them if they are in the minority.

        We use the implementation described in

        Errator: a Tool to Help Detect Annotation Errors in the Universal Dependencies Project
        by Guillaume Wisniewski (LREC 2018) and
        "How Bad are PoS Taggers in Cross-Corpora Settings? Evaluating Annotation Divergence in the UD Project"
        by Guillaume Wisniewski, François Yvon (NAACL 2018)

        It uses generalized suffix trees to find repetitions across sentences which are flagged if the repetitions
        are labeled differently.

        Args:
            sentences: a (num_instances, num_tokens) ragged string sequence containing the text/surface form of each instance
            tags: a (num_instances, num_tokens) ragged string sequence containing the noisy label for each instance
        Returns:
            a (num_samples, num_tokens) ragged boolean array containing the flag for each instance

        """
        corrected, flags = self._compute_variation(sentences, tags)
        return flags

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def supports_correction(self) -> bool:
        return True

    def correct(self, sentences: RaggedStringArray, tags: RaggedStringArray) -> ak.Array:
        corrected, flags = self._compute_variation(sentences, tags)
        return corrected

    def _compute_variation(
        self, sentences: RaggedStringArray, tags: RaggedStringArray
    ) -> Tuple[RaggedStringArray, ak.Array]:
        assert len(sentences) == len(tags)

        tree = Tree()

        corrected = np.empty(len(sentences), dtype=object)
        flags = np.empty(len(sentences), dtype=object)

        sentences = [[x.lower() for x in sentence] for sentence in sentences]

        assert len(sentences) == len(sentences)

        logging.info("Building suffix tree")
        for sentence_id, (cur_sentence, cur_tags) in tqdm(enumerate(zip(sentences, tags)), total=len(sentences)):
            assert len(cur_sentence) == len(cur_tags)

            corrected[sentence_id] = deepcopy(list(cur_tags))
            flags[sentence_id] = [False] * len(cur_sentence)
            tree.add(sentence_id, cur_sentence)

            assert len(corrected[sentence_id]) == len(flags[sentence_id])

        logging.info("Finding variations")
        for count, path in sorted(tree.maximal_repeats(), reverse=True):
            repeated_tokens = list(path.S[path.start : path.end])

            # We ignore repeats that have no context, that is consisting only
            # of a single token
            if len(repeated_tokens) <= 1:
                continue

            # We collect the label sequences for the repeat, if there are more than one, we flag them
            annotations = defaultdict(int)

            # We store sentence id, begin and end for the match so that we can later flag it if necessary
            match_indices = []

            for matched_sentence_id, _ in tree.find_all(repeated_tokens):
                matched_sentence = sentences[matched_sentence_id]
                matched_tags = tags[matched_sentence_id]

                # Our suffix tree does not give us indices when finding all, therefore, we search for them
                # explicitly. It can be that there are duplicate matches in a sentence, which we also
                # handle here.
                for match_begin, match_end in self._find_sublist_indices_in_list(matched_sentence, repeated_tokens):
                    key = tuple(matched_tags[match_begin:match_end])
                    annotations[key] += 1
                    match_indices.append((matched_sentence_id, match_begin, match_end))

            assert len({len(a) for a in annotations}) == 1, "Annotations should all have the same size"

            # If we have different annotations for the same token sequence, then we flag the differences
            # to the most common annotation for that sequence.
            if len(annotations) > 1:
                most_common_annotation = max(annotations, key=annotations.get)

                for matched_sentence_id, match_begin, match_end in match_indices:
                    tags_for_sentence = tags[matched_sentence_id]

                    for i, idx in enumerate(range(match_begin, match_end)):
                        label = tags_for_sentence[idx]
                        mca = most_common_annotation[i]

                        if mca != label:
                            corrected[matched_sentence_id][idx] = mca
                            flags[matched_sentence_id][idx] = True

        corrected = ak.Array(corrected)
        flags = ak.Array(flags)

        return corrected, flags

    @staticmethod
    def _find_sublist_indices_in_list(hay_list: List[Any], needle_list: List[Any]) -> Iterator[Tuple[int, int]]:
        """This returns indices start,end so that hay_list[start:end] == needle_list"""
        assert len(hay_list) > 0
        assert len(needle_list) > 0

        for start_idx in range(0, len(hay_list) - len(needle_list) + 1):
            end_idx = start_idx + len(needle_list)

            sublist = hay_list[start_idx:end_idx]
            assert len(sublist) == len(needle_list)

            if sublist == needle_list:
                yield start_idx, end_idx


class VariationNGramsSpan(Detector):
    """We use the implementation described in

    Inconsistencies in Crowdsourced Slot-Filling Annotations: A Typology and Identification Methods
    by Stefan Larson, Adrian Cheung, Anish Mahendran, Kevin Leach, Jonathan K. Kummerfeld
    Proceedings of the 28th International Conference on Computational Linguistics
    COLING 2020
    """

    def __init__(self, k: int = 1):
        self._k = k

    def score(self, sentences: RaggedStringArray, tags: RaggedStringArray, **kwargs) -> ak.Array:
        """Collects n-grams and their respective label sequences, if there are disagreements, then flag
        them if they are in the minority.

        We use the implementation described in
        Inconsistencies in Crowdsourced Slot-Filling Annotations: A Typology and Identification Methods
        by Stefan Larson, Adrian Cheung, Anish Mahendran, Kevin Leach, Jonathan K. Kummerfeld
        Proceedings of the 28th International Conference on Computational Linguistics
        COLING 2020

        It uses a window of `k` to the left and right of a span, if the window has the same surface form
        but a different label for the span, then we flag it (unless it is the majority label).

        Args:
            sentences: a (num_instances, num_tokens) ragged string sequence containing the text/surface form of each instance
            tags: a (num_instances, num_tokens) ragged string sequence containing the noisy label for each instance
        Returns:
            a (num_samples, num_tokens) ragged boolean array containing the flag for each instance

        """

        corrected, flags = self._compute_variation(sentences, tags)
        return flags

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def supports_correction(self) -> bool:
        return True

    def correct(self, sentences: RaggedStringArray, tags: RaggedStringArray) -> ak.Array:
        corrected, flags = self._compute_variation(sentences, tags)
        return corrected

    def _compute_variation(self, X: RaggedStringArray, y: RaggedStringArray) -> Tuple[RaggedStringArray, ak.Array]:
        entities = []
        surface_forms = []

        assert len(X) == len(y)

        # Compute counts
        counts = defaultdict(lambda: defaultdict(int))

        sentences = [[x.lower() for x in sentence] for sentence in X]

        for sentence, tags in zip(sentences, y):
            assert len(sentence) == len(tags)

            current_entities = parse_spans(tags, SpanEncoding.BIO)
            cur_surface_forms = []

            for span in current_entities:
                ngram_start = max(span.start - self._k, 0)
                ngram_end = min(span.end + self._k, len(tags))

                surface_form = tuple(sentence[ngram_start:ngram_end])
                cur_surface_forms.append(surface_form)

                counts[surface_form][span.type] += 1

            entities.append(current_entities)
            surface_forms.append(cur_surface_forms)

        corrected = []
        flags = []

        # Find annotation errors
        for current_surface_forms, current_entities in zip(surface_forms, entities):
            assert len(current_surface_forms) == len(current_entities)

            new_tags = []
            current_flags = []

            for surface_form, span in zip(current_surface_forms, current_entities):
                counts_for_surface_form = counts[surface_form]
                most_common_label = max(counts_for_surface_form, key=counts_for_surface_form.get)
                label = most_common_label
                new_tags.append(label)

                current_flags.append(label != span.type)

            corrected.append(new_tags)
            flags.append(current_flags)

        corrected = ak.Array(corrected)
        flags = ak.Array(flags)

        return corrected, flags
