import tkinter as tk
from tkinter import messagebox
from tkinter import *
root = tk.Tk()

def main():
    import numpy as np
    import cv2
    import pickle
    import pandas as pd

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    eye_face = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    #profile_face = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    name1=[]
    labels={"person name":1}
    with open("labels.pickle","rb") as f:
        og_labels=pickle.load(f)
        labels={v:k for k,v in og_labels.items()}


    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #face bounding box
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for(x,y,w,h) in faces:
            #print(x,y,w,h)
            roi_gray= gray[y:y+h, x:x+w]
            roi_color= frame[y:y+h, x:x+w]

            id_,conf=recognizer.predict(roi_gray)
            if conf>=45 and conf <=85:
               print(id_)

            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            name1.append(labels[id_])

            color=(255,125,205)
            stroke=2
            cv2.putText(frame, name, (x,y), font, 2, stroke)


            img_item= "my-image.png"
            cv2.imwrite(img_item, roi_gray)


            color= (255, 0, 0)
            stroke= 2
            end_cord_x= x+w
            end_cord_y= y+h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke,cv2.LINE_8)
    #eye bounding box
            eyes=eye_face.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the Capture
    cap.release()
    cv2.destroyAllWindows()

    name2=name1
    NList=[]
    popo=list(og_labels.keys())
    s=set(popo)
    name1=list(dict.fromkeys(name1))
    NList=[x for x in name1 if x in s]
    NList=["PRESENT STUDENTS"]+NList


    filename="attendence"
    df = pd.DataFrame(NList)
    df.to_excel(r'New_attendance.xlsx', index = True)

    t = Text(root)
    for x in NList:
        t.insert(END, x + '\n')
    t.pack()
btn=tk.Button(root,text="Start Attendance",command=main,padx=5,pady=5,relief=GROOVE,width=15)
btn.pack()

def FaceTrain():
    import os
    from PIL import Image
    import numpy as np
    import cv2
    import pickle    #pip install opencv-contrib-python

    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    BASE_DIR =os.path.dirname(os.path.abspath(__file__)) # to find the location of faces_train.py in os
    IMAGE_DIR= os.path.join(BASE_DIR, "images") #to find files called images in same file

    current_id=0
    label_ids={}
    y_labels=[]
    x_tain=[]

    for root,dirs,files in os.walk(IMAGE_DIR):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path=os.path.join(root,file)
                label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
                #print(label,path)
                if not  label in label_ids:
                    label_ids[label]=current_id
                    current_id += 1

                id_=label_ids[label]
                print(label_ids)

                pil_image = Image.open(path).convert("L")  #gray scale
                size = (1024,1024)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array=np.array(final_image, "uint8") #convert image into numbers
                #print(image_array)
                faces= face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x,y,w,h) in faces:
                    roi= image_array[y:y+x , x:x+w ]
                    x_tain.append(roi)
                    y_labels.append(id_)


    #print(y_labels)
    #print(x_tain)

    with open("labels.pickle","wb") as f:
        pickle.dump(label_ids, f)


    recognizer.train(x_tain,np.array(y_labels))
    recognizer.save("trainner.yml")

    messagebox.showinfo("Message","Training complete")

btn2=Button(root,text="Train Faces",width=10,command=FaceTrain)
btn2.place(relx=0.5,rely=0.5,anchor='sw')
btn2.pack()
root.mainloop()


































