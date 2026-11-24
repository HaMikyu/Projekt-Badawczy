import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from helpers import download_single_essay, get_essay_links


LIST_URL = "https://studycorgi.com/year/best-essays-of-2021/"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "essays_2021"
MAX_WORKERS = 8


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    links = get_essay_links(LIST_URL)
    print(f"Znaleziono linkow: {len(links)}")

    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_single_essay, i, url, OUTPUT_DIR): (i, url)
            for i, url in enumerate(links, start=1)
        }
        for done_count, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            if item:
                results.append(item)
            else:
                failed += 1
            if done_count % 25 == 0 or done_count == len(links):
                print(f"Postep: {done_count}/{len(links)} (zapisano: {len(results)}, bledy: {failed})")

    results.sort(key=lambda x: x["index"])
    index_path = OUTPUT_DIR / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "title", "url", "words"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Zakonczono. Pobrano {len(results)} esejow do: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
