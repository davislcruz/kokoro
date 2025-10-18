## Kokoro TTS: Multilingual Text-to-Speech

Kokoro is a modern, PyTorch-based text-to-speech (TTS) system for generating
high-quality, natural-sounding speech from text. It's designed for both
production use and research, emphasizing speed, modularity, and clarity.

## Features at a Glance

### Multilingual and Flexible Voices

  * 9 Supported Languages: English (US/UK), Spanish, French, Hindi, Italian, Brazilian Portuguese, Japanese, and Mandarin Chinese
  * Voice Options: Multiple voices per language with smart defaults. Supports voice blending, custom voice loading, and cached downloads from Hugging Face Hub

### Interfaces and Audio Output

  * CLI and Python API: Simple tools for generating audio from text or files
  * High-Quality Audio: 24 kHz WAV output with optional real-time playback (--speak)
  * Speech Controls: Adjustable speech speed and configurable output paths

### Technical Highlights

  * Advanced Text Processing: Grapheme-to-Phoneme (G2P) conversion, language-specific phonemization, and automatic chunking for long text
  * Neural Architecture: ~82M parameter model combining an ALBERT-based encoder with an iSTFTNet vocoder for natural, expressive speech
  * Performance and Hardware Support: Optimized for CPU, CUDA (GPU), and MPS (Apple Silicon) with automatic device detection
  * Deployment Ready: Supports ONNX export and Triton inference server for scalable production serving
  * Open Source: Apache-licensed weights enable deployment anywhere from production environments to personal projects

### Acknowledgements

  * 🛠️ @yl4579 for architecting StyleTTS 2.
  * 🏆 @Pendrokar for adding Kokoro as a contender in the TTS Spaces Arena.
  * 📊 Thank you to everyone who contributed synthetic training data.
  * ❤️ Special thanks to all compute sponsors.
  * 👾 Discord server: https://discord.gg/QuGxSWBfQy
  * 🪽 Kokoro is a Japanese word that translates to "heart" or "spirit". Kokoro is also a character in the Terminator franchise along with Misaki.

<img src="https://static0.gamerantimages.com/wordpress/wp-content/uploads/2024/08/terminator-zero-41-1.jpg" width="400" alt="kokoro" />
