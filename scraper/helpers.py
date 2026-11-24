import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

STOP_EXACT = {
    "works cited",
    "work cited",
    "references",
    "reference",
    "reference list",
    "bibliography",
    "footnotes",
    "citations",
    "sources",
    "sources cited",
    "list of references",
    "appendix",
    "appendices",
}

PROMO_PHRASES = (
    "we will write a custom",
    "get your 100% customized",
    "custom academic paper",
    "use a 15% discount",
    "study corgi",
    "studycorgi",
    "check out our",
    "check out this",
    "are you about to write",
    "are you looking for an essay",
    "essay sample",
    "essay example",
    "then you are at the right place",
    "here, you’ll find",
    "here, you'll find",
    "here you’ll find",
    "here you'll find",
    "this essay sample is for you",
)


def fetch_html(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=25)
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def get_essay_links(list_url):
    html = fetch_html(list_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links = []
    seen = set()
    for a in soup.select("article h2 a, article h3 a, h2 a, h3 a"):
        href = a.get("href")
        if not href:
            continue
        if "studycorgi.com/" not in href:
            continue
        if "/year/" in href or "/category/" in href or "/tag/" in href:
            continue
        if href not in seen:
            seen.add(href)
            links.append(href)
    return links


def clean_text(text):
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_stop_heading(text):
    low = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    if low in STOP_EXACT:
        return True
    return low.startswith(("works cited", "references", "reference list", "bibliography"))


def is_promo(text):
    low = text.lower()
    return any(phrase in low for phrase in PROMO_PHRASES)


def extract_essay_text(html):
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else ""
    box = soup.select_one(".article__content")
    if not box:
        return title, ""

    children = list(box.find_all(recursive=False))
    toc_index = -1
    for i, child in enumerate(children):
        classes = child.get("class") or []
        if "toc" in classes:
            toc_index = i
            break

    if toc_index != -1:
        children = children[toc_index + 1 :]

    paragraphs = []
    for child in children:
        name = child.name
        if name in ("h2", "h3", "h4"):
            heading = clean_text(child.get_text(" ", strip=True))
            if is_stop_heading(heading):
                break
        elif name == "p":
            txt = clean_text(child.get_text(" ", strip=True))
            if txt and not is_promo(txt):
                paragraphs.append(txt)

    return title, "\n\n".join(paragraphs)


def slug_from_url(url):
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1] if path else "essay"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug)
    return slug[:80]


def save_essay(output_dir, index, url, body):
    slug = slug_from_url(url)
    filename = f"{index:03d}_{slug}.txt"
    path = Path(output_dir) / filename
    path.write_text(body, encoding="utf-8")
    return filename


def download_single_essay(index, url, output_dir):
    html = fetch_html(url)
    if not html:
        return None
    title, body = extract_essay_text(html)
    if len(body.split()) < 30:
        return None
    filename = save_essay(output_dir, index, url, body)
    return {
        "index": index,
        "filename": filename,
        "title": title,
        "url": url,
        "words": len(body.split()),
    }
