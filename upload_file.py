
from dotenv import load_dotenv

import secrets
import string
import boto3
from botocore.exceptions import NoCredentialsError
load_dotenv()
def generate_random_string_id(length=12):
    # Define the alphabet: letters and digits
    alphabet = string.ascii_letters + string.digits
    # Generate a secure random string
    random_id = ''.join(secrets.choice(alphabet) for _ in range(length))
    return random_id
def upload_to_s3(file_name:str,object_name:str,content_type="",bucket_name = 'assets-clusterprotocol'):
    
    if object_name is None:
        object_name = generate_random_string_id()+'.png'

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket_name, object_name,ExtraArgs={'ContentType': content_type})
        object_url = "https://s3-ap-south-1.amazonaws.com/{0}/{1}".format(
    bucket_name,
    object_name)
        print(f"File {file_name} uploaded to {object_url}")
        return {"success":True,"url":object_url}
    except FileNotFoundError:
        print(f"The file {file_name} was not found")
        return {"success":False}
    except NoCredentialsError:
        print("Credentials not available")
        return {"success":False}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"success":False}
