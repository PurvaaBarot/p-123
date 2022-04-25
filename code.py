import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os , ssl , time
import pandas as pd
import cv2

df=pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")
x=np.load("image (1).npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
print("x" , x)
classes=['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J','K', 'L', 'M','N', 'O','P', 'Q', 'R', 'S', 'T','U', 'V', 'W','X', 'Y','Z']
classes

n_classes=len(classes)
n_classes

samples=5

plt.figure(figsize=(n_classes*2,(1+samples*2)))
column_index=0

for p in classes:
  i=np.flatnonzero(y==p)
  i=np.random.choice(i, samples , replace=False)
  row_index=0
  for h in i:
    plot_index=row_index*n_classes+column_index+1
    row_index+=1
  column_index+=1

x_train , x_test , y_train , y_test=train_test_split(x , y , train_size=7500 , test_size=2500)

x_train = x_train/255.0
x_test = x_test/255.0

lr=LogisticRegression(solver="saga" , multi_class="multinomial")
lr.fit(x_train , y_train)

y_pred=lr.predict(x_test)
accuracy_score(y_test , y_pred)

cap=cv2.VideoCapture(0)
while True:
    try:
        ret , frame = cap.read()
        gray=cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)    
        height , width =gray.shape
        upper_left=(int(width/2-56) , int(height/2-56))
        bottom_right=(int(width/2+56) , int(height/2+56))
        cv2.rectangle(gray , upper_left , bottom_right , (0,255,0) , 2)
        roi=gray[upper_left[1] : bottom_right[1] , upper_left[0] : bottom_right[0]]
        im_pil=Image.fromarray(roi)
        image_bw=im_pil.convert("L")
        image_bw_resize=image_bw.resize((22,30) , Image.ANTIALIAS)
        image_bw_resize_inverted=PIL.ImageOps.invert(image_bw_resize)
        pixel_filter=20
        min_pixel=np.percentile(image_bw_resize_inverted , pixel_filter)
        image_bw_resize_inverted_scaled=np.clip(image_bw_resize_inverted-min_pixel,0,255)
        max_picel=np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scaled=np.asarray(image_bw_resize_inverted_scaled)/max_picel
        test_sample=np.array(image_bw_resize_inverted_scaled).reshape(1,660)
        test_pred=lr.predict(test_sample)
        print("Predicted digit is " , test_pred)
        cv2.imshow("Frame" , gray)
        if cv2.waitKey(1) & 0xFF == ord("Q"):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
