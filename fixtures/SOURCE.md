# QASPER demo source

- Paper: David R. So, Chen Liang, and Quoc V. Le, *The Evolved Transformer*, arXiv:1901.11117. The [arXiv HTML](https://arxiv.org/html/1901.11117) identifies the paper's license as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Dataset: [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper), config `qasper`, train split, Dataset Viewer row index `536`. The QASPER dataset card lists CC BY 4.0.
- Retrieval URL: [QASPER Dataset Viewer row 536](https://datasets-server.huggingface.co/rows?dataset=allenai%2Fqasper&config=qasper&split=train&offset=536&length=1). The original run used an untruncated export of this row. The exact paper-text SHA-256 appears in `qasper_demo_manifest.json`.
- Projection: the preparation script reads only the paper information, abstract, and `full_text.section_name`/`full_text.paragraphs` from the selected row. It saves 49 paragraphs with zero-based `sNpM` IDs and reconstructs source quotes from checked spans. No QASPER annotated question, answer, or gold evidence is written into the fixture.
- The repository contains the demo manifest and the offline assessment records. Generate the paper-text fixture locally with `python3 scripts/prepare_qasper_fixture.py`, or pass a local Dataset Viewer rows export via `--rows-json`. The demo question and expected Mock responses are authored for this example.
