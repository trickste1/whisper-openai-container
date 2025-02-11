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

def handler(event, context):
    try:
        #print("Received event: " + json.dumps(event, indent=2))
        print(event['Records'])
        body = urllib.parse.unquote_plus(event['Records'][0]['body'], encoding='utf-8')
        s3key = body.s3key
        print("key:", s3key)
        os.makedirs("/tmp/data", exist_ok=True)
        os.chdir('/tmp/data')

        file_to_transcribe=f"/tmp/data/{s3key}"
        # Downloading file to transcribe
        s3.download_file(bucket, s3key, file_to_transcribe)

        # GPU!! (if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", download_root="/usr/local").to(device)
        #model = whisper.load_model("medium")
        #result = model.transcribe(file_to_transcribe, fp16=False, language='English', verbose=True)
        result = model.transcribe(file_to_transcribe, fp16=False, verbose=True)
        print(result['text'])
        #print(s['text'].strip())
        object = s3.put_object(Bucket=bucket, Key=s3key+'.srt', Body=result["text"].strip())

        # try:
        #     # Generate a pre-signed URL for the S3 object
        #     expiration = 3600  # URL expiration time in seconds
        #     response = s3.generate_presigned_url(
        #         'get_object',
        #         Params={'Bucket': bucket, 'Key': s3key+'.text'},
        #         ExpiresIn=expiration
        #     )

        #     output = f"Transcribed: {key}.text - {response}"

        # except ClientError as e:
        #     print(e)

        return {
            "statusCode": 200,
            "body": json.dumps(result["text"].strip())
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 500,
            "body": json.dumps("Error processing the file")
        }


