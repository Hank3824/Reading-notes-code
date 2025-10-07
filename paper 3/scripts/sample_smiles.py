#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import List, Optional


def reservoir_sample_lines(input_path: str, k: int, seed: Optional[int] = None, has_header: bool = True) -> List[str]:
    if seed is not None:
        random.seed(seed)

    reservoir: List[str] = []
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        header = f.readline() if has_header else None

        for i, line in enumerate(f):
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line

    if has_header and header is not None:
        return [header] + reservoir
    return reservoir


def main():
    parser = argparse.ArgumentParser(description='Reservoir sampling for SMILES dataset')
    parser.add_argument('--input', required=True, help='Input .txt path')
    parser.add_argument('--output', required=True, help='Output .txt path')
    parser.add_argument('--k', type=int, default=1000, help='Number of samples to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_header', action='store_true', help='Indicate input has no header line')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    has_header = not args.no_header
    sampled_lines = reservoir_sample_lines(args.input, args.k, args.seed, has_header=has_header)

    with open(args.output, 'w', encoding='utf-8', newline='') as out:
        for line in sampled_lines:
            out.write(line if line.endswith('\n') else (line + '\n'))

    print(f'Sampled {args.k} lines from {args.input} -> {args.output}')


if __name__ == '__main__':
    main()


