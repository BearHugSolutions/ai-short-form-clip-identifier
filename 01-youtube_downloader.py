#!/usr/bin/env python3
"""
Alternative YouTube Transcript & Video Downloader (using yt-dlp)
More reliable CLI tool to download transcripts and videos from YouTube
"""

import sys
import re
import os
import subprocess
import argparse
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    url = url.strip()
    
    # Handle youtu.be URLs
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    # Handle youtube.com URLs
    if 'youtube.com' in url and 'watch' in url:
        parsed = urlparse(url)
        return parse_qs(parsed.query).get('v', [None])[0]
    
    # If it's already a video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    
    return None


def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sanitize_filename(filename):
    """Remove or replace characters that are invalid in filenames"""
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    return filename


def download_transcript(video_id, base_filename):
    """Download and save transcript files"""
    print(f"Downloading transcript for video ID: {video_id}")
    
    try:
        # Get transcript using the new API
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        
        # Convert to raw data format for easier processing
        transcript = fetched_transcript.to_raw_data()
        
        # Create filenames
        filename_with_timestamps = f"{base_filename}_with_timestamps.txt"
        filename_without_timestamps = f"{base_filename}_transcript.txt"
        
        # Save transcript with timestamps
        with open(filename_with_timestamps, 'w', encoding='utf-8') as f:
            for entry in transcript:
                timestamp = format_time(entry['start'])
                f.write(f"[{timestamp}] {entry['text']}\n")
        
        print(f"✓ Transcript with timestamps saved to: {filename_with_timestamps}")
        
        # Save transcript without timestamps
        with open(filename_without_timestamps, 'w', encoding='utf-8') as f:
            for entry in transcript:
                f.write(f"{entry['text']}\n")
        
        print(f"✓ Transcript without timestamps saved to: {filename_without_timestamps}")
        print(f"✓ Total transcript segments: {len(transcript)}")
        
        # Show preview
        print(f"\nTranscript preview:")
        for i, entry in enumerate(transcript[:3]):
            timestamp = format_time(entry['start'])
            print(f"  [{timestamp}] {entry['text']}")
        if len(transcript) > 3:
            print("  ...")
            
        return True
        
    except Exception as e:
        print(f"✗ Error downloading transcript: {e}")
        print("\nPossible transcript issues:")
        print("- Video has no transcript available")
        print("- Video is private or doesn't exist")
        print("- Transcript is disabled by uploader")
        return False


def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_video_with_yt_dlp(url, base_filename):
    """Download video using yt-dlp (more reliable than pytube)"""
    print(f"\nDownloading video from: {url}")
    
    if not check_yt_dlp():
        print("✗ yt-dlp is not installed!")
        print("Install it with: pip install yt-dlp")
        return False
    
    try:
        # Define output filename template
        output_template = f"{base_filename}_video.%(ext)s"
        
        # yt-dlp command with options for best quality mp4
        cmd = [
            'yt-dlp',
            '--format', 'best[ext=mp4]/best',  # Prefer mp4, fallback to best available
            '--output', output_template,
            '--no-playlist',  # Only download single video even if URL is in playlist
            url
        ]
        
        print("Starting download with yt-dlp...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Find the downloaded file
        expected_files = [f"{base_filename}_video.mp4", f"{base_filename}_video.webm", 
                         f"{base_filename}_video.mkv"]
        downloaded_file = None
        
        for filename in expected_files:
            if os.path.exists(filename):
                downloaded_file = filename
                break
        
        if downloaded_file:
            print(f"✓ Video downloaded: {downloaded_file}")
            file_size = os.path.getsize(downloaded_file) / (1024*1024)
            print(f"✓ File size: {file_size:.1f} MB")
            return True
        else:
            print("✗ Downloaded file not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading video: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        print("\nPossible issues:")
        print("- Video is private, deleted, or region-restricted")
        print("- Age-restricted content")
        print("- Network connection issues")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Alternative YouTube Transcript & Video Downloader (using yt-dlp)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "My Video" "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s "Tutorial" "https://youtu.be/dQw4w9WgXcQ" --transcript-only
  %(prog)s "Lecture" "dQw4w9WgXcQ" --video-only

Prerequisites:
  - Install yt-dlp: pip install yt-dlp
  - Install youtube-transcript-api: pip install youtube-transcript-api
        """
    )
    
    parser.add_argument('filename', help='Base name for output files (without extension)')
    parser.add_argument('url', help='YouTube video URL or video ID')
    parser.add_argument('--transcript-only', action='store_true', 
                       help='Download only transcript files')
    parser.add_argument('--video-only', action='store_true',
                       help='Download only video file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.transcript_only and args.video_only:
        print("Error: Cannot specify both --transcript-only and --video-only")
        sys.exit(1)
    
    # Extract video ID
    video_id = extract_video_id(args.url)
    if not video_id:
        print(f"Error: Could not extract video ID from '{args.url}'")
        sys.exit(1)
    
    # Sanitize filename
    base_filename = sanitize_filename(args.filename)
    if not base_filename:
        print("Error: Invalid filename provided")
        sys.exit(1)
    
    print(f"Using base filename: {base_filename}")
    print(f"Video ID: {video_id}")
    print("-" * 50)
    
    success_count = 0
    total_operations = 0
    
    # Download transcript (unless video-only)
    if not args.video_only:
        total_operations += 1
        if download_transcript(video_id, base_filename):
            success_count += 1
    
    # Download video (unless transcript-only)
    if not args.transcript_only:
        total_operations += 1
        if download_video_with_yt_dlp(args.url, base_filename):
            success_count += 1
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"Download complete: {success_count}/{total_operations} successful")
    
    if success_count == 0:
        print("No files were downloaded successfully.")
        sys.exit(1)
    elif success_count < total_operations:
        print("Some downloads failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("All downloads completed successfully!")


if __name__ == "__main__":
    main()