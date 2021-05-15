import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='RuShiftEval GlossReader - test score calculator')
parser.add_argument('--predictions-filepath', type=str, default='data/preds/result.tsv')
parser.add_argument('--gold-scores-filepath', type=str, default='data/RuShiftEval/post-eval/annotated_testset.tsv')


def calculate_test_score(args):
    preds_df = pd.read_csv(args.predictions_filepath, sep='\t', header=None)
    gold_df = pd.read_csv(args.gold_scores_filepath, sep='\t', header=None)

    correlations = preds_df.corrwith(gold_df, method='spearman')
    correlations.index = ('pre-Soviet:Soviet', 'Soviet:post-Soviet', 'pre-Soviet:post-Soviet')
    mean_correlation = correlations.mean()

    print('Mean correlation:', mean_correlation)
    print('Correlations for all epoch pairs:', correlations.to_string(), sep='\n')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    calculate_test_score(args=args)
