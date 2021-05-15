import os
from utils.train_data_generator import load_json


def load_words(epochs_samples_dir):
    all_words = []
    _, _, filenames = next(os.walk(epochs_samples_dir))

    for filename in filenames:
        current_file = os.path.join(epochs_samples_dir, filename)
        word_examples = load_json(current_file)
        all_words.extend(word_examples)

    return all_words


def construct_test_samples(epochs_samples_dir, large_vectors_path, base_vectors_path):
    all_words = load_words(epochs_samples_dir)
    all_words = {word['id']: word for word in all_words}

    large_preds = load_json(large_vectors_path)
    base_preds = load_json(base_vectors_path)

    large_preds_dict = {pred['id']: pred for pred in large_preds}
    base_preds_dict = {pred['id']: pred for pred in base_preds}
    samples = []

    for pred_id in large_preds_dict.keys():
        large_pred = large_preds_dict[pred_id]
        base_pred = base_preds_dict[pred_id]
        current_sample = {
            'id': pred_id,
            'lemma': all_words[pred_id]['lemma'],
            'base_context_output1': base_pred['context_output1'],
            'base_context_output2': base_pred['context_output2'],
            'large_context_output1': large_pred['context_output1'],
            'large_context_output2': large_pred['context_output2']
        }
        samples.append(current_sample)

    return samples
