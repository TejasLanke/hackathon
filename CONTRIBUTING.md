# Contributing

Use short-lived branches from `main`:

```text
main
|- ml-model
|- gradcam
|- frontend
`- integration
```

Each owner should work mainly in their assigned module, pull or rebase from `main` before significant changes, make small commits, and avoid committing datasets or large model weights. Merge only changes that pass tests.

```powershell
git checkout -b ml-model
git add .
git commit -m "Add EfficientNet training pipeline"
git push -u origin ml-model
```

Before merging, update the branch and resolve conflicts locally:

```powershell
git checkout ml-model
git pull --rebase origin main
pytest
git checkout main
git merge --no-ff ml-model
```
