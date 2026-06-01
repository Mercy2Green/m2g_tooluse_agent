Local demo videos for M2G ToolUse.

To view locally:

1. Start a simple HTTP server in the `videos` folder (Python 3):

```bash
cd m2g_tooluse_agent/videos
python3 -m http.server 8080
```

2. Open http://localhost:8080 in your browser and click `index.html`.

Notes:
- Videos are large; serve over HTTP rather than opening `file://` to avoid cross-origin playback issues.
- If a browser refuses autoplay, use the player controls to start playback.
