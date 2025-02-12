import os
import shutil
import json
import urllib3
import urllib.parse
import whisper
import torch
import boto3
import warnings


s3 = boto3.client("s3")
bucket = "explainer-create-films-dev"

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def format_srt(segments):
    srt_output = []
    for i, segment in enumerate(segments, start=1):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        srt_output.append(f"{i}\n{format_time(start_time)} --> {format_time(end_time)}\n{text}\n")

    return "\n".join(srt_output)


def handler(event, context):
    try:
        print("Received event: " + json.dumps(event, indent=2))
        # print(event)
        body = event['body']
        s3key = body.get('s3key')
        print("key:", s3key)
        os.makedirs("/tmp/data", exist_ok=True)
        os.chdir('/tmp/data')

        file_to_transcribe = "/tmp/data/test.mp4"
        print("file_to_transcribe:", file_to_transcribe)
        print(bucket, s3key, file_to_transcribe)
        # Downloading file to transcribe
        s3.download_file(bucket, s3key, file_to_transcribe)
        print("downloaded file from s3")
        # GPU!! (if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("defined device", device)
        model = whisper.load_model("base", download_root="/usr/local").to(device)
        print("loaded model")
        #model = whisper.load_model("medium")
        #result = model.transcribe(file_to_transcribe, fp16=False, language='English', verbose=True)
        result = model.transcribe(file_to_transcribe, fp16=False, verbose=True)

        # whisper.utils.write_srt(result, "/tmp/transcription.srt")
        srt_text = format_srt(result["segments"])
        detected_language = result["language"]
        print(result['text'])

        return {
            "statusCode": 200,
            "body": json.dumps({
                "srtText": srt_text,
                "detectedLang": detected_language
            })
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps("Error processing the file")
        }


