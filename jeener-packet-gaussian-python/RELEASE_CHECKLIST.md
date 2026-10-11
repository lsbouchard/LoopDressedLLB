# Release Checks

Run from the repository root:

```bash
MPLBACKEND=Agg python -m unittest discover -s jeener-packet-gaussian-python/tests
make -C jeener-packet-gaussian-c test
git diff --check
```

Before publishing:

1. Review `AUDIT.md`, including the unresolved paper null control.
2. Confirm that current regression values match the documented aggregation.
3. Check the version and authors in both `CITATION.cff` files.
4. Rebuild the PDF manual when its LaTeX source changes.
5. Keep the non-commercial license and as-is, no-support terms with both codebases.
6. Review the commit diff and platform test results before merging.

The historical source is preserved by Git history. A matching source hash or
agreement between two implementations is not a substitute for mathematical
control tests.
