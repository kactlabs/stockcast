"""Find and remove duplicate stock symbols in stocks.txt."""


def deduplicate(path: str = "stocks.txt") -> None:
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    seen = set()
    dupes = []
    clean = []
    for symbol in lines:
        if symbol in seen:
            dupes.append(symbol)
        else:
            seen.add(symbol)
            clean.append(symbol)

    if dupes:
        print(f"Duplicates found: {', '.join(dupes)}")
        with open(path, "w") as f:
            f.write("\n".join(clean) + "\n")
        print(f"Removed {len(dupes)} duplicate(s). {len(clean)} symbols remaining.")
    else:
        print("No duplicates found.")


if __name__ == "__main__":
    deduplicate()
