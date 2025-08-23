# Troubleshooting

## "module diffusers has no attribute WanPipeline"

Use the registry loader:

```python
DiffusionPipeline.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16)
```

## "VAE shape mismatch"

Ensure the video VAE is used and `vae/config.json` contains `base_dim`, `z_dim`, `scale_factor_spatial`, and `scale_factor_temporal`.

## "Tokenizers version conflict"

`transformers==4.49.0` requires `tokenizers==0.21.x`. Install a compatible version:

```bash
pip install "tokenizers>=0.21,<0.22"
```

