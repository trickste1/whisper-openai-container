import os
import json
import torch
import boto3
import stable_whisper

s3 = boto3.client("s3")
bucket = "explainer-create-films-dev"

def handler(event, context):
    try:
        print("Received event: " + json.dumps(event, indent=2))
        # print(event)
        # parsedEvent = json.loads(event)
        body = event['body']
        print("Received body: " + body)
        parsedBody = json.loads(body)
        s3key = parsedBody["s3key"]
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
        model = stable_whisper.load_model("base", download_root="/usr/local").to(device)
        print("loaded model")
        #model = whisper.load_model("medium")
        #result = model.transcribe(file_to_transcribe, fp16=False, language='English', verbose=True)
        result = model.transcribe(file_to_transcribe, fp16=False, verbose=True)

        # whisper.utils.write_srt(result, "/tmp/transcription.srt")
        # srt_text = format_srt(result["segments"])
        # print(result['text'])
        detected_language = result.language
        srt_text = result.to_srt_vtt(segment_level=True ,word_level=False)
        print("RESULT")
        print(srt_text)
        print("DETECTED LANG")
        print(result.language)
        # print(detected_language)

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


