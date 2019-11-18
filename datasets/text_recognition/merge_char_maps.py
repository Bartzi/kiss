import argparse
import json


def main(args):
    chars = set()
    new_metadata = {}

    for map in args.char_maps:
        with open(map) as f:
            char_map = json.load(f)

        metadata = char_map['metadata']
        for key, value in metadata.items():
            if char_map[str(value)] not in chars:
                new_metadata[key] = char_map[str(value)]

        del char_map['metadata']
        chars = chars.union(set(char_map.values()))

    print(new_metadata)
    new_char_map = {i: char for i, char in enumerate(chars)}
    new_reverse_char_map = {char: i for i, char in new_char_map.items()}
    for key, value in new_metadata.items():
        new_metadata[key] = new_reverse_char_map[value]

    new_char_map['metadata'] = new_metadata

    with open(args.destination, 'w') as f:
        json.dump(new_char_map, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that takes multiple char map json files and merges them together")
    parser.add_argument("char_maps", nargs='+', help="char maps to merge")
    parser.add_argument("-d", "--destination", required=True, help="destination path where merged char map shall be saved")

    args = parser.parse_args()
    main(args)
