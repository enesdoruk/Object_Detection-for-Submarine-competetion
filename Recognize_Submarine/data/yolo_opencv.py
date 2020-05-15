import cv2
import argparse
import numpy as np
import time

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to config file', default="D:\\ENES\\TEKNOFEST\\cemberden_gecme\\data\\yolo-obj.cfg")
ap.add_argument('-w', '--weights', 
                help = 'path to pre-trained weights', default="D:\\ENES\\TEKNOFEST\\cemberden_gecme\\data\\yolo-obj_last.weights")
ap.add_argument('-cl', '--classes', 
                help = 'path to objects.names',default="D:\\ENES\\TEKNOFEST\\cemberden_gecme\\data\\objects.names")
args = ap.parse_args()



def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

window_title= "BTU_ROV"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)


classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


net = cv2.dnn.readNet(args.weights,args.config)


cap = cv2.VideoCapture(1)


while cv2.waitKey(1) < 0 or False:
    
    hasframe, image = cap.read()
    image=cv2.resize(image, (416, 416)) 
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    print(len(outs))
    
    
    
    for out in outs: 
        print(out.shape)
        for detection in out:
            
        
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
		
		
		
		
		
        ekran_merkez_x = 213
        ekran_merkez_y = 213
        cerceve_sol_x = np.int(x)
        cerceve_sag_x = np.int(x+w)
        cerceve_alt_y = np.int(y)
        cerceve_üst_y = np.int(y+h)  
		
		
		
		
		
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
		
		
		
		
		
		
		
		
		
		
        cv2.circle(image,(213,213),5,(255,0,0),-1)
        
        
        
	
       	cv2.circle(image,(int(x+(w/2)),int(y+(h/2))),5,(0,255,0),-1)
        
        
        
        
				
        cv2.line(image,(213,213),(int(x+(w/2)),int(y+(h/2))),(0,0,255),5)
		
		
			
		
        if cerceve_sag_x > ekran_merkez_x > cerceve_sol_x and  cerceve_üst_y > ekran_merkez_y > cerceve_alt_y: 
            print("çerceve sınırlarında")
            print("sadece ileri")
            time.sleep(10)
    		
    		
    		
        if cerceve_sag_x > ekran_merkez_x > cerceve_sol_x and cerceve_alt_y > ekran_merkez_y: 
            print("nesnenin alt tarafındasın")
            print("üste doğru hareket et")
            time.sleep(10)
    		
    		
    		
        if cerceve_sag_x > ekran_merkez_x > cerceve_sol_x and ekran_merkez_y > cerceve_üst_y: 
            print("nesnenin üstündesin")
            print("alçal")
            time.sleep(10)
    
    		
        if ekran_merkez_x > cerceve_sag_x and cerceve_üst_y > ekran_merkez_y > cerceve_alt_y:
            print("nesnenin sağ tarafındasın")
            print("sola doğru hareket et")
            time.sleep(10)
    
    			
        if ekran_merkez_x < cerceve_sol_x and cerceve_üst_y > ekran_merkez_y > cerceve_alt_y:
            print("nesnenin sol tarafındasın")
            print("sağa doğru hareket et")
            time.sleep(10)
    			
        if ekran_merkez_x < cerceve_sol_x and cerceve_alt_y > ekran_merkez_y:
            print("nesnenin sol alt tarafındasın")
            print("sağ üst tarafa ilerle")
            time.sleep(10)
    			
        if ekran_merkez_x <cerceve_sol_x and cerceve_üst_y < ekran_merkez_y:
            print("nesne cemberin sol üstünde")
            print("sağ alta doğru hareket et")
            time.sleep(10)
    			
    			
        if  ekran_merkez_x > cerceve_sag_x and cerceve_alt_y > ekran_merkez_y:
            print("nesnenin sağ altındasın")
            print("sol üste doğru hareket et")
            time.sleep(10)
    			
    			
        if ekran_merkez_x > cerceve_sag_x and cerceve_üst_y < ekran_merkez_y:
            print("nesnenin sag üstündesin")
            print("sol alta doğru hareket et")
            time.sleep(10)
			
			

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0))
    
    cv2.imshow(window_title, image)
