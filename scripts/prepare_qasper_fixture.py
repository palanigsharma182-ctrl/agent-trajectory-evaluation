"""Rebuild the demo fixture from one QASPER Dataset Viewer row."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_fixture(manifest: dict, response: dict) -> dict:
    metadata = manifest["source"]
    rows = [item for item in response["rows"] if str(item["row"]["id"]) == manifest["paper"]["id"]]
    if len(rows) != 1 or rows[0]["row_idx"] != metadata["row_index"] or rows[0].get("truncated_cells"):
        raise ValueError("The input must contain the exact, untruncated QASPER demo row.")
    row = rows[0]["row"]
    full_text = row["full_text"]
    paragraphs = [
        {"id": f"s{section_index}p{paragraph_index}", "section": section, "text": text}
        for section_index, (section, texts) in enumerate(zip(full_text["section_name"], full_text["paragraphs"], strict=True))
        for paragraph_index, text in enumerate(texts)
        if text.strip()
    ]
    content = {"abstract": row["abstract"], "paragraphs": paragraphs}
    if sha256(content) != metadata["content_sha256"]:
        raise ValueError("Paper content differs from the reviewed snapshot. Use the matching QASPER row export.")

    by_id = {item["id"]: item["text"] for item in paragraphs}
    propositions = []
    for item in manifest["propositions"]:
        start, end = item["span"]
        quote = by_id[item["paragraph_id"]][start:end]
        if not quote or sha256(quote) != item["quote_sha256"]:
            raise ValueError(f"Source span does not match for {item['id']}.")
        propositions.append({**{key: value for key, value in item.items() if key not in {"span", "quote_sha256"}}, "quote": quote})
    return {"paper": {**manifest["paper"], **content}, "task": manifest["task"], "propositions": propositions,
            "relations": manifest["relations"], "experiment": manifest["experiment"], "demo": manifest["demo"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-json", type=Path, help="Local Dataset Viewer rows export containing the demo paper")
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures/qasper_paper.json")
    args = parser.parse_args()
    manifest = load(ROOT / "fixtures/qasper_demo_manifest.json")
    if args.rows_json:
        response = load(args.rows_json)
    else:
        source = manifest["source"]
        query = urlencode({"dataset": source["dataset"], "config": source["config"], "split": source["split"],
                           "offset": source["row_index"], "length": 1})
        try:
            with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=30) as stream:
                response = json.load(stream)
        except OSError as exc:
            parser.exit(1, f"Cannot fetch the QASPER demo row: {exc}\nRetry or supply a local export via --rows-json.\n")
    fixture = build_fixture(manifest, response)
    if args.output.exists() and load(args.output) != fixture:
        parser.error(f"Existing output differs: {args.output}. Choose another --output path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
