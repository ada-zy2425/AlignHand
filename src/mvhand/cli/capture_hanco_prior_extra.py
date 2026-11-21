"""Capture frozen-prior outputs for the HanCo training and development groups that are still missing."""
import argparse
import json
from pathlib import Path
from mvhand.cli.capture_prior import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    run(json.loads(parser.parse_args().config.read_text()),hanco_a3=True)


if __name__=='__main__':main()
