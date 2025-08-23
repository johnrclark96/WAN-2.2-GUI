## Summary

<!-- Describe the changes -->

## Testing

```text
ruff check .
mypy --ignore-missing-imports wan_ps1_engine.py
python -m compileall -q .
python wan_ps1_engine.py --help
python wan_ps1_engine.py --dry-run --attn auto --mode t2v --frames 8 --width 512 --height 288 --model_dir models/WAN
pytest -q
```

```
<paste logs here>
```

`[RESULT] OK` or `[RESULT] FAIL`

