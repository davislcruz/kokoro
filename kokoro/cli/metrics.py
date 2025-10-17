"""Performance metrics formatting for Kokoro TTS CLI."""


def format_metrics(metrics: dict, detailed: bool = False) -> str:
    # Determine performance category
    rtf = metrics['rtf']
    if rtf < 1.0:
        performance_status = "✅ Faster than real-time"
    elif rtf == 1.0:
        performance_status = "⚖️  Real-time"
    else:
        performance_status = "❌ Slower than real-time"

    if detailed:
        lines = [
            "\n" + "=" * 50,
            "PERFORMANCE METRICS",
            "=" * 50,
            f"Text length:      {metrics['text_length']:,} characters",
            f"Chunks generated: {metrics['chunk_count']}",
            f"Synthesis time:   {metrics['synthesis_time']:.2f} s",
            f"Audio duration:   {metrics['audio_duration']:.2f} s",
            f"Total samples:    {metrics['total_samples']:,}",
            f"RTF:              {metrics['rtf']:.4f}",
            f"Performance:      {performance_status}",
            "=" * 50
        ]
        return "\n".join(lines)
    else:
        return (
            f"\n{performance_status}\n"
            f"Generated {metrics['audio_duration']:.2f}s audio in "
            f"{metrics['synthesis_time']:.2f}s (RTF: {metrics['rtf']:.3f})"
        )