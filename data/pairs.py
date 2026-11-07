import argparse


def iter_pairs(manifest):
    """Yield (image_path, caption) pairs from a manifest file."""
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            image_path, caption = line.rstrip("\n").split("\t", 1)
            yield image_path, caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.parse_args()
