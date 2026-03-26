"""
Update versions.json. Adds a record based on the passed version. Derived from script
used in Folium.

Usage:

python update_version_json.py --version 'v0.1.0'
"""

import argparse
import json


def main():
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Update version.json")
    parser.add_argument(
        "--version", "-v", required=True, type=str, help="The new version to add"
    )
    args = parser.parse_args()
    version = args.version

    # Open the JSON
    with open("versions.json") as f:
        version_file = json.load(f)


    # Clean up the stable alias from previous release
    for entry in version_file:
        entry["aliases"] = []

    version_entry = {
        "version": version,
        "title": version,
        "aliases": ["stable"],
    }
    # add after dev
    version_file.insert(1, version_entry)

    # Write the updated file
    with open("versions.json", "w") as f:
        json.dump(version_file, f, indent=2)


if __name__ == "__main__":
    main()
