#importing all necessary libraries
import numpy as np
import tensorflow as tf
import cv2
from imutils.video import VideoStream
import imutils

#defining a function that will do all the calculations to get predictions 
#on a image like where is a face in th image and on that face whether 
#there is a mask or not and returning these predictions from the functione

def get_detections(image, face_dectector, mask_detector):
    # from image getting a blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
            (104.0, 177.0, 123.0))
    #a blob is a modified contour in a way
    #basicaly it is group of connected pixels that share some same properties

    #Inputing this blob into face_detector 
    face_dectector.setInput(blob)   
    
    #and getting predictions from it by forward() method that returns a 4D blob 
    predictions = face_dectector.forward()
    print(predictions.shape)
    
    #getting height and width from the image where shape has(h, w, channels)
    (h, w) = image.shape[:2]

    #getting 3 empty lists to save faces , their locations 
    # and prediction in an image
    faces_in_one_image = []
    locations =[]
    predictions_list = []   #it will have mask_detactor predictions

    #getting best prediction out of all predictions 
    #that face-detector model has given and from that prediction
    #getting face locations in that particular prediction that 
    #has highest probablity 
    for prediction_number in range(0, predictions.shape[2]):

        #every prediction that corresponds to a prediction_number has 7 items in it
        #out of these 7 items probablity is on 3rd number(index=2)
        probablity = predictions[0,0,prediction_number,2]
        #print("04")
        #getting probablities that has value > 0.5
        #so by doing this all low probablites will be removed
        if probablity > 0.5:
            #out of those 7 items co-ordinates of detected face
            #is on from 4th postion to 7th position(index -> 3,4,5,6)
            #but these co-ordinates values are in form of fractions
            #of h,w of image, so getting co-ordinates in exact x,y positions
            start_x = max(0, int(predictions[0,0,prediction_number,3] * w))
            start_y = max(0, int(predictions[0,0,prediction_number,4] * h))
            end_x = min(w-1, int(predictions[0,0,prediction_number,5] * w))
            end_y = min(h-1, int(predictions[0,0,prediction_number,6] * h))
             

            #now that i have co-ordinates
            #getting face from the image with these co-ordinates
            #because cv2 is used on thids opretion image wil be in BGR
            #cropping image in the co-ordinates
            face = image[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            #resizing this face into 224*224 and preprocessing it
            # so that it can be feeded into mobilenetV2
            face = cv2.resize(face, (224,224))

            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            #adding this face into faces list
            faces_in_one_image.append(face)

            #adding co-ordinates of above image into loactions list as tuple
            locations.append((start_x, start_y, end_x, end_y))

        #now if any face is found with probablity > 0.5
        # that is if any face is found in faces_in_one_image list
        # then feeding it to mask_detector model
        # for that first converting faces into numpy array  
    if len(faces_in_one_image) > 0:
        faces_in_one_image = np.array(faces_in_one_image, dtype="float32")
        predictions_list = mask_detector.predict(faces_in_one_image, batch_size=20)
        
    #returning the locations and prediction_list for one image
    return(locations, predictions_list)  


#loading face_detector caffe model(weights_file) from the directory where it is saved
#to load it prototxt file of that caffe model is also required so 
#a directory where thr file is located
caffe_path = "face_detector\res10_300x300_ssd_iter_140000.caffemodel"
deploy_path =  "face_detector\deploy.prototxt"



face_detector = cv2.dnn.readNet(deploy_path, caffe_path)

#loading the mask detectotr model from the directory where it is saved
mask_detector = tf.keras.models.load_model("mask_detector.model")


#getting the video camera open of device
live_cam = VideoStream(src=0).start()

#getting each frame(image) from the live video
#and getting and showing predictions on theses frames(images)
while True:
    #reading every frame from video
    #and resising frame width
    image = live_cam.read()
    image = imutils.resize(image, width=400)
     
    #getting predictions and face locations in this image   
    (locations, prediction_list) = get_detections(image, face_detector, mask_detector)
    
    #getting location and prediction of each face in this image
    for(loc, p) in zip(locations, prediction_list):
        
        #getting co-ordinates of each face
        (start_x, start_y, end_x, end_y) = loc

        #getting probablities of mask and without mask for each face
        (mask, withoutmask) = p


        #a colored rectangle will be drawn upon all faces in this image
        #and a tag with info about mask or without mask will be
        #written upon that box, 
        #so getting that tag abd color of box
        tag = " "
        color = (0,0,0)
        if mask>withoutmask:
            tag = "mask " + str(mask*100)[ :5]
            color = (0, 255, 0)     #green color
        else:
            tag = "no mask " + str(withoutmask*100)[ :5]        
            color = (0, 0 ,255)       #red color
        
        
        #putting a rectange on each face in image
        cv2.rectangle(image, (start_x, start_y), (end_x,end_y), color, 1)

        #putting tag on each rectangle in the image    
        cv2.putText(image, tag, (start_x, start_y - 15), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, color, 1)
    
    #displaying the each image from video
    cv2.imshow("image", image)

    #to exit press e
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break
cv2.destroyAllWindows()
live_cam.stop
        





