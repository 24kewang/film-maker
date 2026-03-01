Phase 1 (LLM + Image Gen)          Phase 2 (Image Gen)           Phase 3 (Video Gen)
┌─────────────────────────┐        ┌──────────────────┐          ┌──────────────────┐
│ 1. Script (Claude)      │        │ 6. Keyframe imgs │          │ 7. 10× LTX-Video │
│ 2. Char descriptions    │        │    (FLUX + CLIP  │          │    (parallel on  │
│ 3. Env description      │        │     conditioning)│          │     Modal A100s) │
│ 4. Keyframe descriptions│        │                  │          │ 8. ffmpeg concat │
│ 5. Reference images     │        │                  │          │                  │
│    (FLUX on Modal)      │        │                  │          │                  │
└──────────┬──────────────┘        └────────┬─────────┘          └────────┬─────────┘
           │                                │                             │
     USER SELECTS                     USER SELECTS                   FINAL FILM
     reference variants               keyframe variants              (MP4 download)