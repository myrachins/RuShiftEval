import json


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def _construct_data_samples(samples_data_path, samples_gold_path):
    data_json = load_json(samples_data_path)
    labels_json = load_json(samples_gold_path)

    labels_dict = {sample['id']: sample for sample in labels_json}
    for sample in data_json:
        sample['score'] = labels_dict[sample['id']]['score']

    return data_json


def construct_train_samples(samples_data_path, samples_gold_path, large_vectors_path, base_vectors_path):
    large_preds = load_json(large_vectors_path)
    base_preds = load_json(base_vectors_path)
    samples = _construct_data_samples(samples_data_path, samples_gold_path)

    large_preds_dict = {pred['id']: pred for pred in large_preds}
    base_preds_dict = {pred['id']: pred for pred in base_preds}

    for sample in samples:
        large_pred = large_preds_dict[sample['id']]
        base_pred = base_preds_dict[sample['id']]

        sample['base_context_output1'] = base_pred['context_output1']
        sample['base_context_output2'] = base_pred['context_output2']

        sample['large_context_output1'] = large_pred['context_output1']
        sample['large_context_output2'] = large_pred['context_output2']

    return samples
