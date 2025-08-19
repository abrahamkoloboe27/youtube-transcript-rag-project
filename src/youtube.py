from urllib.parse import urlparse, parse_qs
from .loggings import configure_logging
logger = configure_logging(log_file="youtube.log", logger_name="__youtube__")

def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url: The YouTube URL.

    Returns:
        The video ID.

    Raises:
        ValueError: If the URL is not a valid YouTube URL.

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=123456")
        '123456'
        >>> extract_video_id("https://youtu.be/123456")
        '123456'
        >>> extract_video_id("https://www.youtube.com/embed/123456")
        '123456'
    """
    logger.info(f"Extracting video ID from URL: {url}")
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if 'youtu.be' in netloc:
        return parsed.path.lstrip('/')
    qs = parse_qs(parsed.query)
    if 'v' in qs:
        return qs['v'][0]
    parts = parsed.path.split('/')
    if 'embed' in parts:
        return parts[-1]
    logger.error(f"Impossible d'extraire l'ID depuis: {url}")

def save_txt(fetched_transcript, out_path='transcript.txt'):
    """
    Saves the fetched transcript to a text file.

    Args:
        fetched_transcript: The fetched transcript.
        out_path: The path to the output text file.

    Returns:
        None
    """
    logger.info(f"Saving TXT to {out_path}")
    with open("./downloads/"+out_path, 'w', encoding='utf-8') as f:
        for seg in fetched_transcript:
            f.write(seg.text.strip() + '\n')
    logger.info(f"Saved TXT to ../downloads/{out_path}")

def save_srt(fetched_transcript, out_path='transcript.srt'):
    """
    Saves the fetched transcript to an SRT file.

    Args:
        fetched_transcript: The fetched transcript.
        out_path: The path to the output SRT file.

    Returns:
        None
    """
    logger.info(f"Saving SRT to {out_path}")
    def fmt_time(s):
        """
        Formats a time in seconds to the SRT time format.

        Args:
            s: The time in seconds.

        Returns:
            The formatted time.
        """
        logger.info(f"Formatting time {s} to SRT format")
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"
    logger.info(f"Formatting SRT file {out_path}")

    with open(out_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(fetched_transcript, start=1):
            logger.info(f"Formatting segment {i}")
            start = seg.start
            duration = seg.duration if hasattr(seg, 'duration') else 3.0
            end = start + duration
            f.write(f"{i}\n")
            f.write(f"{fmt_time(start)} --> {fmt_time(end)}\n")
            f.write(seg.text.strip() + "\n\n")
    logger.info(f"Saved SRT to {out_path}")

# def main(video_url: str, lang='en'):
#     """
#     Main function to extract the video ID, fetch the transcript, and save it in both TXT and SRT formats.

#     Args:
#         video_url: The YouTube URL.
#         lang: The language code for the transcript.

#     Returns:
#         None
#     """
#     vid = extract_video_id(video_url)
#     # récupère la meilleure disponible (auto-generated fonctionne)
#     ytt = YouTubeTranscriptApi()
#     transcript = ytt.fetch(video_id=vid, languages=[lang])
#     save_txt(transcript, out_path=f"{vid}.txt")
#     save_srt(transcript, out_path=f"{vid}.srt")
