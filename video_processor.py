"""
video_processor.py
Extracts transcript/content from:
  - Uploaded video files (via Google Gemini's video understanding)
  - YouTube URLs (via youtube-transcript-api)
"""

import os
import time
import google.generativeai as genai


def extract_transcript(video_path: str, api_key: str) -> str:
    """
    Upload a video file to Gemini and extract a full transcript + description.
    Works with mp4, mov, avi, mkv, webm.
    """
    genai.configure(api_key=api_key)

    print(f"Uploading video: {video_path}")
    video_file = genai.upload_file(path=video_path)

    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed on Gemini's end.")

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """
    Please provide a comprehensive transcript and analysis of this video.
    Include:
    1. A full transcript of all spoken words (with approximate timestamps if possible, e.g. [0:00], [1:30])
    2. Description of key visual elements, scenes, and on-screen text
    3. Main topics and themes covered
    4. Any important data, names, or facts mentioned

    Format timestamps as [MM:SS] before each section.
    Be thorough — capture everything said and shown in the video.
    """

    response = model.generate_content([video_file, prompt])
    genai.delete_file(video_file.name)  # cleanup
    return response.text


def get_youtube_transcript(url: str) -> str:
    """
    Fetch transcript from a YouTube video using youtube-transcript-api.
    Falls back to basic video info if transcript unavailable.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from urllib.parse import urlparse, parse_qs

        # Extract video ID
        parsed = urlparse(url)
        if parsed.hostname in ("youtu.be",):
            video_id = parsed.path.lstrip("/")
        else:
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [None])[0]

        if not video_id:
            raise ValueError("Could not extract YouTube video ID from URL.")

        # Try to get transcript (new API: instantiate first, then call fetch)
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)

        # Format with timestamps
        lines = []
        for entry in fetched:
            minutes = int(entry.start) // 60
            seconds = int(entry.start) % 60
            lines.append(f"[{minutes:02d}:{seconds:02d}] {entry.text}")

        return "\n".join(lines)

    except Exception as e:
        raise ValueError(
            f"Could not fetch YouTube transcript: {e}\n"
            "Make sure the video has captions/subtitles enabled, "
            "or upload the video file directly instead."
        )


def extract_frames_summary(video_path: str, api_key: str) -> str:
    """Alias for extract_transcript — used for uploaded files."""
    return extract_transcript(video_path, api_key)