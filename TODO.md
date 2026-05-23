# TODO

- [x] Identify root cause of `ModuleNotFoundError: sklearn`.
- [x] Update `requirements.txt` to include `scikit-learn`.
- [x] Install dependencies.
- [x] Run `python src/components/data_ingestion.py` and observe failures.
- [ ] Fix dataset path handling in `src/components/data_ingestion.py` so it reads `notebook/data/stud.csv` reliably.
- [ ] Re-run `python src/components/data_ingestion.py`.
- [ ] Verify logs are written to `logs/<timestamp>.log` with the expected INFO lines.
- [ ] Verify `artifacts/train.csv`, `artifacts/test.csv`, and `artifacts/raw.csv` are created.

