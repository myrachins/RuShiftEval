import numpy as np
from scipy import stats
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


class VectorsDotPredictor:
    def __init__(self, normalize=True, norm_ord=2):
        self.normalize = normalize
        self.norm_ord = norm_ord

    def predict(self, out_vector_1, out_vector_2):
        out_vector_1 = np.array(out_vector_1)
        out_vector_2 = np.array(out_vector_2)

        if self.normalize:
            out_vector_1 /= np.linalg.norm(out_vector_1, ord=self.norm_ord)
            out_vector_2 /= np.linalg.norm(out_vector_2, ord=self.norm_ord)

        return np.sum(out_vector_1 * out_vector_2)


class VectorsDistPredictor:
    def __init__(self, normalize=True, norm_ord=2):
        self.normalize = normalize
        self.norm_ord = norm_ord

    def predict(self, out_vector_1, out_vector_2):
        out_vector_1 = np.array(out_vector_1)
        out_vector_2 = np.array(out_vector_2)

        if self.normalize:
            out_vector_1 /= np.linalg.norm(out_vector_1, ord=self.norm_ord)
            out_vector_2 /= np.linalg.norm(out_vector_2, ord=self.norm_ord)

        return np.linalg.norm(out_vector_1 - out_vector_2, ord=self.norm_ord)


def get_default_predictors():
    predictors = [
        VectorsDotPredictor(normalize=True, norm_ord=2),
        VectorsDotPredictor(normalize=True, norm_ord=1),
        VectorsDotPredictor(normalize=False),
        VectorsDistPredictor(normalize=True, norm_ord=2),
        VectorsDistPredictor(normalize=False, norm_ord=2),
        VectorsDistPredictor(normalize=False, norm_ord=1),
        VectorsDistPredictor(normalize=True, norm_ord=1),
    ]
    return predictors


def _construct_dataset(samples, predictors):
    x, y = [], []

    for sample in samples:
        current_features = []

        for predictor in predictors:
            current_features.append(predictor.predict(
                sample['base_context_output1'], sample['base_context_output2']
            ))
            current_features.append(predictor.predict(
                sample['large_context_output1'], sample['large_context_output2']
            ))

        x.append(current_features)
        y.append(sample['score'])

    return np.array(x), np.array(y)


def train_linear_model(samples, predictors, val_size, random_seed=42):
    x, y = _construct_dataset(samples, predictors)
    mask = y != 0
    x, y = x[mask], y[mask]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size, random_state=random_seed)

    lr = Ridge(random_state=random_seed)
    lr_params = {
        'alpha': np.linspace(0, 2, 100)
    }
    model = GridSearchCV(lr, lr_params)
    model.fit(x_train, y_train)

    y_val_preds = model.predict(x_val)
    val_corr = stats.spearmanr(y_val, y_val_preds)

    return model, val_corr


def _sample_test_data(test_sample, predictors):
    current_features = []

    for predictor in predictors:
        current_features.append(predictor.predict(
            test_sample['base_context_output1'], test_sample['base_context_output2']
        ))
        current_features.append(predictor.predict(
            test_sample['large_context_output1'], test_sample['large_context_output2']
        ))

    return current_features


def predict(model, test_samples, predictors):
    test_data = []

    for sample in test_samples:
        sample_data = _sample_test_data(sample, predictors)
        test_data.append(sample_data)

    scores = model.predict(test_data)
    word_to_scores = defaultdict(list)

    for i, sample in enumerate(test_samples):
        current_lemma = sample['lemma']
        word_to_scores[current_lemma].append(scores[i])

    word_to_score = {word: np.mean(scores) for word, scores in word_to_scores.items()}

    return word_to_score

