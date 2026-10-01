# Package F handoff: interactive replay client

## Files

- `web/` contains a Three.js client with orbit, pan, and zoom controls; a fly and keyboard scene; monitor texture; replay transport; key activity/history; and a sampled activity display.
- `web/package.json` and `web/package-lock.json` pin the npm client dependencies.
- Vite builds browser-ready assets into `web/` so the Python server can serve that directory directly at `/`.

## Run

From the repository root:

```sh
cd web
npm install
npm run build
cd ..
uv run python -m backend.server
```

Then open `http://127.0.0.1:8000/`. During frontend development, `cd web && npm run dev` starts Vite; its `/api/replay` fetch targets the same origin, so use the Python server as the API host or configure a Vite proxy for a separate backend port.

The client populates the target selector from `GET /api/targets`, requests the selected replay from `GET /api/replay?target_id=…`, and reports exact, syntax, and elaboration status separately. The client requests `GET /api/replay` and consumes `{metadata:{target_id,mode},layout:{keys:[...]},events:[...],final_text,target}`. Playback is event driven and can be paused, stepped, replayed, scrubbed, and speed adjusted. Selecting a history row moves the replay cursor to the corresponding event. Only `type: key` events contribute characters. The monitor and side output use the latest key event’s cumulative `text` buffer when available, with a key-character reconstruction fallback, and render it identically.

## Build result

`npm install` completed successfully (15 packages audited, zero vulnerabilities). `npm run build` completed successfully with Vite 7.3.6. Vite reports the expected large Three.js bundle warning (about 523 kB minified / 132 kB gzip) and an `outDir` warning because output is intentionally written into `web/` for direct static serving. The output is self-contained and no longer requests fonts or scripts from a CDN. Build output is `web/index.html` plus `web/assets/`.

## Limits and schema notes

The brain panel is a generated point cloud that colors sampled points from event `activity`; it does not load MaleCNS/FlyWire topology or represent measured dynamics. Labels identify activity as synthetic/modelled and state that topology is unavailable and the policy is scripted. Scene dimensions are a visualization mapping that places backend key and foot XY on the same viewer surface and maps backend Z to viewer height. A moving kinematic foot marker and line show the reported `foot_pose`; contact onset and key events depress and highlight the corresponding key. Event `body` pose is used to position the fly at the inspected tick. The frontend does not create or score contacts.
