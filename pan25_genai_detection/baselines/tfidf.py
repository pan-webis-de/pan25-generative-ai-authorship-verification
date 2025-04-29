# Copyright 2025 Janek Bevendorff, Webis
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

from pathlib import Path
import pickle
import typing as t

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from pan25_genai_detection.baselines.detector_base import DetectorBase

__all__ = ['TfidfDetector']


class TfidfDetector(DetectorBase):
    """
    Baseline LLM detector using TF-IDF features and an SVM.
    """

    _BASEDIR = Path(__file__).parent

    def _normalize_scores(self, scores):
        # Optimise c@1
        scores[np.abs(scores - .5) < .05] = 0.5
        return scores

    def _get_score_impl(self, text: t.Iterable[str]) -> np.ndarray:
        clf, vec = pickle.load((self._BASEDIR / 'tfidf_model.pkl').open('rb'))
        return clf._predict_proba_lr(vec.transform(text))[:, 1]

    def _predict_impl(self, text: t.Iterable[str]):
        return self._normalize_scores(self._get_score_impl(text)) >= 0.5

    def _predict_with_score_impl(self, text: t.Iterable[str]) -> t.Tuple[np.ndarray, np.ndarray]:
        s = self._get_score_impl(text)
        return s, self._normalize_scores(s) >= 0.5

    @classmethod
    def train(cls, jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)

        print('Fitting vectorizer...')
        vec = TfidfVectorizer(ngram_range=(1, 4), max_features=1000)
        vec.fit(df['text'])

        print('Training SVM...')
        clf = LinearSVC()
        clf.fit(vec.transform(df['text']), df['label'])

        pickle.dump((clf, vec), (cls._BASEDIR / 'tfidf_model.pkl').open('wb'))
