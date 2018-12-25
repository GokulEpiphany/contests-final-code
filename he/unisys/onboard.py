import boto3

s3 = boto3.resource('s3')

images=[('gokul-1.jpg', 'Mr Gokul M')
       ]

for image in images:
  file = open(image[0],'rb')
  object = s3.Object('deep-guard-images','Rekognition-Images/'+ image[0])
  ret = object.put(Body=file, Metadata={'FullName':image[1]})
