# How to run Ollama with parallel slots on macOS

## Problem
The Ollama macOS app runs with `-np 1` (1 parallel slot). Multiple AI agents queue up and wait, making runs very slow.

## Fix: Run `ollama serve` manually with `OLLAMA_NUM_PARALLEL`

### Step 1: Disable Ollama auto-start
1. Go to **System Settings → General → Login Items & Extensions**
2. Under **Allow in the Background**, turn **OFF** Ollama

### Step 2: Quit the Ollama app
```bash
osascript -e 'tell application "Ollama" to quit'
```

### Step 3: Stop the Ollama launch agent
```bash
launchctl bootout gui/$(id -u) /Applications/Ollama.app/Contents/Library/LaunchAgents/com.ollama.ollama.plist
pkill -9 -f "ollama"
```

### Step 4: Start Ollama with parallel slots
```bash
OLLAMA_NUM_PARALLEL=4 /Applications/Ollama.app/Contents/Resources/ollama serve > /tmp/ollama.log 2>&1 &
```

### Step 5: Verify
```bash
# Wait for model to load
ollama run gemma4:12b "hi"

# Check parallel slots
ps aux | grep "llama-server" | grep -v grep | grep -o "\-np [0-9]*"
# Should show: -np 4
```

## To go back to normal
Re-enable Ollama in Login Items and reboot, or:
```bash
pkill -f "ollama serve"
open -a Ollama
```

## Notes
- `OLLAMA_NUM_PARALLEL=4` with gemma4:12b (7.6GB) uses ~10-12GB RAM total (model + 4 KV caches). Safe on 32GB Mac.
- Don't set higher than 4 — diminishing returns and more RAM usage.
- This only works when running `ollama serve` from terminal, NOT the Ollama app.
