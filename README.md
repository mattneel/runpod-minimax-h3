![MiniMax H3 Singularity — one model to rule them all](https://raw.githubusercontent.com/huchukato/runpod-minimax-h3/main/media/banner-singularity.jpg)

# MiniMax H3 Turbo — ComfyUI Serverless Worker

[![Runpod](https://api.runpod.io/badge/huchukato/runpod-minimax-h3)](https://console.runpod.io/hub)

A production-ready [Runpod Serverless](https://docs.runpod.io/serverless/overview) worker running **MiniMax H3** (Comfy-Org INT8 ConvRot) on ComfyUI — text/image/first-last-frame/reference-to-video **with audio**, multi-scene **storyboards via the Director**, turbo LoRAs, TensorRT RIFE + upscaling, wildcards and an uncensored Qwen3-VL enhancer. Built for [ForgeHub](https://github.com/huchukato/ForgeHub), but the endpoint speaks plain Runpod API.

## Deploy

1. **Storage first**: create a [network volume](https://console.runpod.io/user/storage) (≥160GB) in a datacenter with 96GB Pro GPUs — e.g. `US-NE-1`.
2. Click **Deploy** → endpoint → **Edit** → set **Max workers = 1** + attach your volume. Cold start is heavy (~100GB of models on first boot, big model loads after), so one warm worker with jobs queued beats N cold workers — Runpod's deploy default is 3, lower it. Workers auto-download the models on first boot (once). No volume = every job fails on missing models.
3. Call it via the Runpod API or ForgeHub below — workflows, wildcards, job queue and outputs included; paste your API key and it detects the endpoint by itself.

[![Download ForgeHub](https://raw.githubusercontent.com/huchukato/runpod-minimax-h3/main/media/forgehub-release.jpeg)](https://github.com/huchukato/ForgeHub/releases/latest)

## Workflows included

| Workflow | Description |
| --- | --- |
| `MiniMaxH3-Turbo-T2VA.json` | Text → video + audio |
| `MiniMaxH3-Turbo-I2VA.json` | Image → video + audio |
| `MiniMaxH3-Turbo-FL2VA.json` | First/last frame → video + audio |
| `MiniMaxH3-Turbo-R2VA.json` | Reference images → video + audio |
| `MiniMaxH3-Singularity-R2VA.json` | Singularity fusion — up to 3 ref images + ref video |
| `MiniMaxH3-Director-Native.json` | **Director**: character sheet + one idea per line → multi-scene storyboard with audio; the worker auto-analyzes the subject and expands each idea via the built-in QwenVL |
| `MiniMaxH3-Director-Storyboard.json` | **Director**: one idea (+ up to 3 storyboard frames) → single video; the in-graph QwenVL turns the idea into a full R2VA prompt |

## Requirements

- **GPU**: 96GB class (`BLACKWELL_96` pool). The 32B text encoder + two video DiTs + TensorRT engines need the headroom.
- **Network volume**: ≥160GB at `/runpod-volume`. Models auto-download from `models-manifest.txt` (~100GB). Disable with `VOLUME_AUTOPOPULATE=false`.
- **CUDA**: 13.0+ hosts.

## Model stack (auto-provisioned)

- DiTs: Comfy-Org `minimax_h3_fl2va` INT8, `Minimax-h3_Singularity_ref2va` INT8 (T2V/I2V/R2V/V2V in one unet)
- Text encoder: `qwen3vl_32b_heretic_minimax_h3_nvfp4` (Momoking)
- Turbo LoRAs: lightx2v fl2v/ref2v 8-step · Style LoRAs: `h3-realism-people` (fal), `h3_character_swap` (akatz)
- VAEs: video fp16 + audio fp32, `taeh3` preview · Face detailer: `face_yolov8m` + `sam_vit_b`
- Enhancer LLM: Qwen3.5-9B Defiant-Fable NEO-MAX Q6_K GGUF + mmproj (also drives the Director's auto-Analyze/Enhance)

## Job input

```json
{
  "input": {
    "workflow": "MiniMaxH3-Director-Native.json",
    "prompt": "she turns from the window, hair catching the golden light\nshe walks toward the camera, playful smile",
    "global_prompt": "warm cinematic realism; her look stays the same in every shot",
    "seconds": 10,
    "images": ["<character sheet>"],
    "enhance": true,
    "video_size": "1344x768 (16:9)",
    "upscale": true,
    "rife": true
  }
}
```

- **Director Native**: `prompt` = one idea per line (each = a segment, `seconds` split evenly); `global_prompt` = style prose; `subject_description` optional (auto-Analyze when empty); `enhance` = per-segment QwenVL autoprompting (default true)
- **Director Storyboard**: `prompt` = one idea (timestamps welcome); `images` = up to 3 storyboard frames; `vl_preset` = `MiniMax › Director Storyboard` / `Director Prose`
- Classic: `config` presets (`native`/`native_turbo`/`r2va_singularity`/`r2va_singularity_turbo`/`10eros`/`10eros_turbo`) swap DiT/LoRA/steps/sampler/shift; `lora` = style LoRA on Singularity (`none`/`realism`/`char_swap`)
- `10eros_turbo`: TenStrip's 10Eros Max TURBO-hybrid beta5 (FL2VA finetune, INT8 ConvRot, turbo fused in, so the turbo LoRA slot is zeroed): 8 steps, euler/simple, shift 12/3. Use it with the T2VA / I2VA / FL2VA workflows. `10eros` (20 steps) expects a non-turbo 10Eros checkpoint, which the manifest does not download.
- `video_size` — `auto` (follows the image/stills) or an explicit MMH3 native preset (`1344x768 (16:9)`, `1184x864 (4:3)`, …)
- `upscale` / `rife` — TensorRT post chain · `"action": "health"` — status + workflow list, no generation

## Reference audio + six-part prompts (ref2v)

Two extra job fields for MiniMaxH3-Turbo-R2VA.json / MiniMaxH3-Singularity-R2VA.json:

- `audios`: up to 3 reference audio files (URL or base64; mp3/wav/flac/ogg/m4a) wired into the
  ref2v node as `<Audio 1>`, `<Audio 2>`, ... Mark a song section `fully_copy` in the prompt to
  have the clip use it exactly (the character performs to it), or `reference` for timbre only.
- `lipsync_audio` (+ optional `lipsync_at`, seconds): the exact soundtrack — a song section or a dub.
  It is pinned onto the output's audio track (MiniMaxH3AddGuide) so the clip plays this audio and
  the picture, mouths included, is generated to it. Set `seconds` to the track's length; past the
  end of the track the model makes up its own audio. `audios` alone is only a reference the model
  may or may not copy.
- `seconds` takes any number, decimals included (8.5). The frame count rounds up to H3's
  17k+5 grid (8 s = 192 frames exactly, 8.5 s = 209 frames = 8.7 s), so for a pinned track pick
  `seconds` at or just under its length. A job that saves no file now fails with ComfyUI's reason
  instead of returning an empty `outputs` list.
- `output: "volume"`: write the result straight onto the network volume instead of returning it inline.
  Files over the inline limit (10 MB) go there automatically. The output entry carries
  `volume_key` (`outputs/<job id>/<file>`); read it through the volume's S3 API:
  `aws s3 cp s3://<volume id>/<volume_key> . --region <DC> --endpoint-url https://s3api-<dc>.runpod.io`.
  Copies are deleted after 24 hours. Set `OUTPUT_S3_BUCKET` on the endpoint to get a ready `s3_uri`.
  RunPod's volume S3 API has no presigned URLs, so `s3Config` pointing at it returns `s3_uri` only;
  other S3 providers also get a presigned `url`.
- `{"action": "fetch", "key": "<volume_key>", "offset": 0}`: read a volume output back without S3 keys,
  8 MiB of base64 per job (`chunk_b64`, `end`, `size`, `done`); repeat with `offset = end` until `done`.
- Any media field (`images`, `audios`, `videos`, `lipsync_audio`, ...) also takes `"volume:<volume_key>"`, an earlier
  job's output still on the volume, so passes can be chained without re-uploading.
- `images`: beyond the workflow's own image loaders, extra images go straight into the ref2v node (up to 9).
- `videos`: up to 3 reference videos (`<Video 1>`, ...), re-timed to 24 fps and capped at 15 s;
  `{"url": ..., "use_audio": true}` also passes the clip's soundtrack as an `<Audio N>`
  (soundtracks are numbered before `audios`).
- `loras`: `[{"name": "char_swap", "strength": 1.0}]` stacks LoRAs from the volume after the workflow's own
  (`char_swap` = Akatz Character Swap V1.5, `char_swap_v1`, `realism`, or part of a file name).
- `config: r2va_stock` / `r2va_stock_turbo`: the stock Comfy-Org ref2va model (what Character Swap was trained
  on); turbo follows the Akatz recipe (FL2V turbo 8-step LoRA, 8 steps, er_sde, normal).
- `sections`: MiniMax's six-part full-reference prompt as an object
  (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
  `overall_soundscape`, `non_diegetic_music`). It is assembled in MiniMax's format and sent
  verbatim (`enhance` is forced off so QwenVL does not rewrite it). Use instead of `prompt`.

```json
{"input": {
  "workflow": "MiniMaxH3-Turbo-R2VA.json",
  "config": "r2va_singularity_turbo",
  "images": ["https://.../singer.png"],
  "audios": ["https://.../chorus.mp3"],
  "seconds": 8,
  "sections": {
    "subject_definitions": "<Subject 1> is the singer in <Picture 1>.\n<Audio 1> is the song section <Subject 1> sings.",
    "summary": "[reference generation + audio reuse] <Subject 1> sings the chorus on a neon rooftop.",
    "retention_analysis": "<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, jacket.\n<Audio 1>: fully_copy - the song is used exactly as given.",
    "detailed_description": "Cinematic music-video look.\n[Shot 1] <Subject 1> (S1) sings to camera, <d>[English] the chorus lyrics</d>",
    "overall_soundscape": "N/A",
    "non_diegetic_music": "N/A"
  }
}}
```

Custom nodes baked in

`ComfyUI-VideoHelperSuite` · `ComfyUI-Easy-Use` · `ComfyUI-Impact-Pack` · `ComfyUI-Impact-Subpack` · `ComfyUI-QwenVL-Mod` · `ComfyUI-MiniMaxH3-Director-Enhanced` · `ComfyUI-RIFE-TensorRT-Auto` · `ComfyUI-Upscaler-TensorRT-Auto` · `ComfyUI-TagForge` · `ComfyUI-PerfectVideoResolution` · `ComfyUI-Pixaroma` · `ComfyUI-KJNodes`

## Companion stack

- 🖼 Image sibling: [huchukato/runpod-qwen21](https://github.com/huchukato/runpod-qwen21) (Qwen Image 2.1 T2I/Edit + Pony)
