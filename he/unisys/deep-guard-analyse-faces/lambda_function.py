import boto3
import io
import datetime
import uuid
import urllib
import os
from PIL import Image
from io import BytesIO
from pprint import pprint

rekognition = boto3.client('rekognition', region_name='us-east-1')
dynamodb = boto3.client('dynamodb', region_name='us-east-1')
dbRekognitionCollection = 'deep-guard-rekognition-collection'
logTable = 'deep-guard-facial-history'
lastPersonTable = 'deep-guard-last-person'
snsArn = 'arn:aws:sns:us-east-1:321442707160:NotifyMe'
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    
    logDateTime = datetime.datetime.now()
    sDateTime = logDateTime.strftime("%d/%m/%Y %H:%M:%S.%f")

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key_name = urllib.unquote_plus(
        event['Records'][0]['s3']['object']['key'].encode('utf8'))
    
    obj = s3.Object(bucket_name=bucket, key=key_name)
    obj_body = obj.get()['Body'].read()
    image = Image.open(BytesIO(obj_body))

    # stream = s3.object.read()
    # # image = Image.open("group1.jpg")
    stream = io.BytesIO()
    image.save(stream,format="JPEG")
    image_binary = stream.getvalue()

    response = rekognition.detect_faces(
        Image={'Bytes': image_binary})
    print(str(response))
    # response = rekognition.detect_faces(
    #     Image={
    #       "S3Object": {
    #         "Bucket": bucket, 
    #         "Name": key_name }
    #           }                                        
    #         )
        
    all_faces=response['FaceDetails']
    
    # Initialize list object
    boxes = []
    
    # Get image diameters
    image_width = image.size[0]
    image_height = image.size[1]
    print('Image Width: ' + str(image_width) + ', Image Height: ' + str(image_height) + '.')
       
    # Crop face from image
    for face in all_faces:
        box=face['BoundingBox']
        x1 = int(box['Left'] * image_width) * 0.9
        y1 = int(box['Top'] * image_height) * 0.9
        x2 = int(box['Left'] * image_width + box['Width'] * image_width) * 1.10
        y2 = int(box['Top'] * image_height + box['Height']  * image_height) * 1.10
        image_crop = image.crop((x1,y1,x2,y2))
        
        stream = io.BytesIO()
        image_crop.save(stream,format="JPEG")
        image_crop_binary = stream.getvalue()
    
        # Submit individually cropped image to Amazon Rekognition
        response = rekognition.search_faces_by_image(
                CollectionId='deep_guard_collection',
                Image={'Bytes':image_crop_binary},
                MaxFaces=1                                     
                )
        
        if len(response['FaceMatches']) > 0:
            # Return results
            print ('Coordinates ', box)
            for match in response['FaceMatches']:
                    
                face = dynamodb.get_item(
                    TableName=dbRekognitionCollection,               
                    Key={'RekognitionId': {'S': match['Face']['FaceId']}}
                    )
        
                if 'Item' in face:
                    person = face['Item']['FullName']['S']
                else:
                    person = 'no match found'
                
                print (match['Face']['FaceId'],match['Face']['Confidence'],person)
                similarity = match['Similarity']
                confidenceScore = match['Face']['Confidence']
                uniqueId = str(uuid.uuid4())
                dynamodb.put_item(TableName=logTable,
                    Item={
                        'UniqueId': {'S': uniqueId},
                        'DateTime': {'S': sDateTime},
                        'FullName': {'S': person},  
                        'Similarity': {'N': str(similarity)}
                    })
                dynamodb.update_item(TableName=lastPersonTable,
                Key={'UniqueId': {'S': '1'}},
                UpdateExpression="set FullName = :r",
                ExpressionAttributeValues={
                    ':r': {'S': person}}
                )
    
        else:
            print('Face not found in collection')
            uniqueId = str(uuid.uuid4())
            # log to db
            dynamodb.put_item(TableName=logTable,
                Item={
                    'UniqueId': {'S': uniqueId},
                    'DateTime': {'S': sDateTime},
                    'FullName': {'S': 'An unknown person'}
                    })
            dynamodb.update_item(TableName=lastPersonTable,
                Key={'UniqueId': {'S': '1'}},
                UpdateExpression="set FullName = :r",
                ExpressionAttributeValues={
                    ':r': {'S': 'An unknown person'}}
                )

            fileName = 'https://s3.amazonaws.com/' + os.environ["unknown_images_bucket"] + '/' + os.environ["unknown_images_folder"] + '/'+ uniqueId +'.jpeg'
            object = s3.Object(os.environ["unknown_images_bucket"], os.environ["unknown_images_folder"] + '/'+ uniqueId +'.jpeg')
            ret = object.put(Body=image_crop_binary)
            
            # Publish SNS notification
            message = 'Unknown person detected.  You can view the image at ' + fileName
            client = boto3.client('sns')
            response = client.publish(
                TargetArn=snsArn,
                Message=message,
                Subject='Unknown Person Detected',
                MessageStructure='raw'
                )

    