#!/usr/bin/env python3
"""Example script demonstrating the local model manager usage."""

from kokoro.local_model_manager import LocalModelManager, get_config, get_model, get_voice
import json

def main():
    print("🚀 Kokoro Local Model Manager Example")
    print("=" * 50)

    # Initialize manager
    manager = LocalModelManager()

    # Check what's available locally FIRST (no downloads)
    print("\n🔍 Checking local availability...")
    print(f"Config available locally: {'✅' if manager.has_local_config() else '❌'}")
    print(f"Model available locally: {'✅' if manager.has_local_model() else '❌'}")
    local_voices = manager.list_local_voices()
    print(f"Local voices: {local_voices if local_voices else '❌ None'}")

    # Method 1: Using the manager class
    print("\n📁 Method 1: Using LocalModelManager class")

    # Get config (downloads ONLY if missing)
    print("\n📋 Getting configuration...")
    config = manager.get_config()
    print(f"✅ Config loaded with {len(config)} keys")

    # Get model (downloads ONLY if missing)
    print("\n🤖 Getting model...")
    model_path = manager.get_model()
    print(f"✅ Model available at: {model_path}")

    # Get a voice (downloads ONLY if missing)
    print("\n🎤 Getting voice...")
    voice_path = manager.get_voice('af')  # Afrikaans voice
    print(f"✅ Voice available at: {voice_path}")

    # Method 2: Using convenience functions
    print("\n📁 Method 2: Using convenience functions")

    # These do the same thing but are more concise
    config2 = get_config()
    model_path2 = get_model()
    voice_path2 = get_voice('en-us')  # English US voice

    print(f"✅ Config keys: {list(config2.keys())[:5]}...")
    print(f"✅ Model: {model_path2.name}")
    print(f"✅ Voice: {voice_path2.name}")

    # Show storage information
    print("\n📊 Storage Information")
    print("-" * 30)
    info = manager.get_storage_info()

    print(f"Base path: {info['base_path']}")
    print(f"Repository: {info['repo_id']}")
    print(f"\nConfig files: {info['config']['files']} ({info['config']['size_mb']:.1f} MB)")
    print(f"Model files: {info['models']['files']} ({info['models']['size_mb']:.1f} MB)")
    print(f"Voice files: {info['voices']['files']} ({info['voices']['size_mb']:.1f} MB)")

    # List available voices
    voices = manager.list_local_voices()
    if voices:
        print(f"\nAvailable voices: {', '.join(voices)}")
    else:
        print("\nNo voices downloaded yet")

    # Advanced usage: Smart voice management - check before downloading
    print("\n🎭 Smart voice management...")
    voice_names = ['ar', 'fr', 'de']  # Arabic, French, German

    for voice_name in voice_names:
        if manager.has_local_voice(voice_name):
            print(f"✅ {voice_name}: Already available locally, using existing file")
        else:
            print(f"📥 {voice_name}: Not found locally, downloading...")
            try:
                voice_path = manager.get_voice(voice_name)
                print(f"✅ {voice_name}: Downloaded to {voice_path.name}")
            except Exception as e:
                print(f"❌ {voice_name}: {e}")

    # Demonstrate local-first behavior
    print("\n🔄 Demonstrating local-first behavior...")
    print("Requesting 'af' voice again - should use existing local file:")
    voice_path = manager.get_voice('af')  # Should use existing file
    print(f"Used existing file: {voice_path}")

    # Final storage info
    print("\n📊 Final Storage Summary")
    print("-" * 30)
    final_info = manager.get_storage_info()
    total_size = (
        final_info['config']['size_mb'] +
        final_info['models']['size_mb'] +
        final_info['voices']['size_mb']
    )
    total_files = (
        final_info['config']['files'] +
        final_info['models']['files'] +
        final_info['voices']['files']
    )

    print(f"Total files: {total_files}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Available voices: {len(final_info['voices']['available_voices'])}")

if __name__ == "__main__":
    main()