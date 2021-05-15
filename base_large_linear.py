import argparse
import random

from utils.train_data_generator import construct_train_samples
from utils.regressor import train_linear_model, predict, get_default_predictors
from utils.test_data_generator import construct_test_samples

parser = argparse.ArgumentParser(description='RuShiftEval GlossReader - large + base + linear regression')
parser.add_argument('--rand-seed', type=int, default=42)
parser.add_argument('--train-val-size', type=float, default=0.1)

parser.add_argument('--rusemshift-1-data', type=str,
                    default='data/RuSemShift/annotations-1/source/dev.rusemshift_1.data')
parser.add_argument('--rusemshift-1-gold', type=str,
                    default='data/RuSemShift/annotations-1/source/dev.rusemshift_1.gold')
parser.add_argument('--rusemshift-1-base-vectors', type=str,
                    default='data/RuSemShift/annotations-1/vectors/dev.vectors.base.rusemshift_1.data')
parser.add_argument('--rusemshift-1-large-vectors', type=str,
                    default='data/RuSemShift/annotations-1/vectors/dev.vectors.large.rusemshift_1.data')

parser.add_argument('--rusemshift-2-data', type=str,
                    default='data/RuSemShift/annotations-2/source/dev.rusemshift_2.data')
parser.add_argument('--rusemshift-2-gold', type=str,
                    default='data/RuSemShift/annotations-2/source/dev.rusemshift_2.gold')
parser.add_argument('--rusemshift-2-base-vectors', type=str,
                    default='data/RuSemShift/annotations-2/vectors/dev.vectors.base.rusemshift_2.data')
parser.add_argument('--rusemshift-2-large-vectors', type=str,
                    default='data/RuSemShift/annotations-2/vectors/dev.vectors.large.rusemshift_2.data')

parser.add_argument('--epochs-samples-dir-12', type=str, default='data/RuShiftEval/epoch-12/samples/')
parser.add_argument('--epochs-base-vectors-12', type=str, default='data/RuShiftEval/epoch-12/base_vectors.json')
parser.add_argument('--epochs-large-vectors-12', type=str, default='data/RuShiftEval/epoch-12/large_vectors.json')

parser.add_argument('--epochs-samples-dir-13', type=str, default='data/RuShiftEval/epoch-13/samples/')
parser.add_argument('--epochs-base-vectors-13', type=str, default='data/RuShiftEval/epoch-13/base_vectors.json')
parser.add_argument('--epochs-large-vectors-13', type=str, default='data/RuShiftEval/epoch-13/large_vectors.json')

parser.add_argument('--epochs-samples-dir-23', type=str, default='data/RuShiftEval/epoch-23/samples/')
parser.add_argument('--epochs-base-vectors-23', type=str, default='data/RuShiftEval/epoch-23/base_vectors.json')
parser.add_argument('--epochs-large-vectors-23', type=str, default='data/RuShiftEval/epoch-23/large_vectors.json')

parser.add_argument('--output-predictions-path', type=str, default='data/preds/result.tsv')


def generate_predictions(args):
    print('Loading train samples...')
    train_samples_1 = construct_train_samples(
        args.rusemshift_1_data, args.rusemshift_1_gold,
        args.rusemshift_1_large_vectors, args.rusemshift_1_base_vectors
    )
    train_samples_2 = construct_train_samples(
        args.rusemshift_2_data, args.rusemshift_2_gold,
        args.rusemshift_2_large_vectors, args.rusemshift_2_base_vectors
    )
    train_samples = train_samples_1 + train_samples_2
    random.seed(args.rand_seed)
    random.shuffle(train_samples)

    print('Training regression...')
    predictors = get_default_predictors()
    model, val_corr = train_linear_model(train_samples, predictors, args.train_val_size, random_seed=args.rand_seed)

    print('Loading test samples for 1-2 epochs pair...')
    test_samples_12 = construct_test_samples(
        args.epochs_samples_dir_12, args.epochs_large_vectors_12, args.epochs_base_vectors_12
    )
    print('Loading test samples for 1-3 epochs pair...')
    test_samples_13 = construct_test_samples(
        args.epochs_samples_dir_13, args.epochs_large_vectors_13, args.epochs_base_vectors_13
    )
    print('Loading test samples for 2-3 epochs pair...')
    test_samples_23 = construct_test_samples(
        args.epochs_samples_dir_23, args.epochs_large_vectors_23, args.epochs_base_vectors_23
    )

    print('Calculating final predictions for 1-2 epochs pair...')
    scores_12 = predict(model, test_samples_12, predictors)
    print('Calculating final predictions for 1-3 epochs pair...')
    scores_13 = predict(model, test_samples_13, predictors)
    print('Calculating final predictions for 2-3 epochs pair...')
    scores_23 = predict(model, test_samples_23, predictors)

    print(f'Saving predictions to "{args.output_predictions_path}"...')
    with open(args.output_predictions_path, 'w', encoding='utf-8') as f:
        for word, score_12 in scores_12.items():
            f.write(f'{word}\t{score_12}\t{scores_23[word]}\t{scores_13[word]}\n')

    print('Predictions were saved!')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    generate_predictions(args=args)
