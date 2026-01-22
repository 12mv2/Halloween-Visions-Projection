# Contributing

1. Fork or clone the repo
2. Create a branch: `git checkout -b your-feature`
3. Make changes, test locally
4. Push and open a Pull Request

## Testing on Xavier

At denhac, pull your branch to Xavier and test:
```bash
cd ~/MLVisions
git fetch origin
git checkout your-feature
python3 games/YourGame/game.py
```

## Platform Notes

- `best.onnx` - Linux/Mac/Windows
- `best_xavier.onnx` - Jetson only (opset 12)

Don't commit IDE configs, `.pyc`, or large binaries.
