"""Command line entry points for the offline demo scorecard."""
from __future__ import annotations

import argparse
import json

from kernsignal.reference import make_demo


def main(argv=None):
    ap = argparse.ArgumentParser(prog='kernsignal', description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo-reference', help='Generate expressly simulated demo data')
    demo.add_argument('--output', default='data/demo_reference.csv')
    demo.add_argument('--respondents', type=int, default=3200)
    demo.add_argument('--seed', type=int, default=41)

    scorer = sub.add_parser('score-panels', help='Generate an independent validation scorecard')
    scorer.add_argument('--human', required=True)
    scorer.add_argument('--synthetic', required=True)
    scorer.add_argument('--output-dir', default='artifacts/scorecard')
    scorer.add_argument('--tvd-threshold', type=float, default=.15)

    args = ap.parse_args(argv)
    if args.command == 'demo-reference':
        result = make_demo(args.output, n=args.respondents, seed=args.seed)
    elif args.command == 'score-panels':
        from kernsignal.validation import score_panels
        result = score_panels(args.human, args.synthetic, args.output_dir,
                              diverge_threshold=args.tvd_threshold)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == '__main__':
    main()
