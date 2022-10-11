import argparse
import json
import pathlib
import random
from os.path import join


def sample(data_dir: str, output_dir: str, ratio: float) -> None:
    sids = []

    with open(join(data_dir, 'train-bv.json'), encoding='utf-8') as reader:
        train_samples = json.load(reader)

    for sample in train_samples:
        for question in sample['questions']:
            sids.append((question['uid'], sample['table']['uid']))
    
    ids = list(range(len(sids)))
    random.shuffle(ids)
    ids = ids[:int(len(ids) * ratio)]

    qids, tids = [], set()

    for i in ids:
        qids.append(sids[i][0])
        tids.add(sids[i][1])

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(join(output_dir, 'train-bv-%.3f.json' % ratio), 'w', encoding='utf-8') as writer:
        json.dump(qids, writer)


def main(args: argparse.Namespace):
    if args.command == 'sample':
        sample(args.data_dir, args.output_dir, args.ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument('ratio', type=float)
    parser.add_argument('--data-dir', type=str, default='data/tatqa/tagop/annotated')
    parser.add_argument('--output-dir', type=str, default='data/tatqa/tagop/reduced')
    args = parser.parse_args()
    main(args)
