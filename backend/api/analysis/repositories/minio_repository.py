import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class MinioRepository:
    def __init__(self, endpoint_url, access_key, secret_key, bucket_name):
        self.s3_client = boto3.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'  # Change if needed
        )
        self.bucket_name = bucket_name
        self._create_bucket_if_not_exists()

    def _create_bucket_if_not_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    def upload_file(self, file_path, object_name):
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            return f"{self.bucket_name}/{object_name}"
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            return None
        
    def upload_file_directly(self, uploaded_file, object_name):
        try:
            # Use 'put_object' instead of 'upload_file' for streaming
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_name,
                Body=uploaded_file
            )
            return f"{self.bucket_name}/{object_name}"
        except ClientError as e:
            logger.error(f"Failed to upload file: {e}")
            return None

    def download_file(self, object_name, file_path):
        try:
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
        except ClientError as e:
            logger.error(f"Failed to download file: {e}")

    def delete_file(self, object_name):
        self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
        
        
    def delete_files(self, object_names):
        # Create a list of objects to delete
        objects_to_delete = [{'Key': object_name} for object_name in object_names]

        # Delete all objects in a single request
        response = self.s3_client.delete_objects(
            Bucket=self.bucket_name,
            Delete={
                'Objects': objects_to_delete,
                'Quiet': True  # Set to False if you want a response for each object deletion
            }
        )

    def list_files(self):
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            return [item["Key"] for item in response.get("Contents", [])]
        except ClientError as e:
            logger.error(f"Failed to list files: {e}")
            return []
