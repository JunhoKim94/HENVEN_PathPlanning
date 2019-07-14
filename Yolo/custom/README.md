HOW-TO

1. Separate train and test dataset through /Yolo/Split_Train_Test.py
2. Fill /Yolo/Custom/object.names with needed classes in order with the image names. E.g 0 -> first class
3. Set right paths in /Yolo/Custom/trainer.data
4. Edits needed for the YOLO cfg files:
    - Classes - num of needed classes
    - Max batches = (classes*2000). eg: 3 classes -> max_batches = 6000
    - Filters=(classes + 5)x3
    - Anchors --> ./darknet detector calc_anchors custom/trainer.data -num_of_clusters 6 -width 416 -height 416




Training:
Testing: 
./darknet detector test custom/trainer.data custom/yolov3.cfg backup/[filename].weights [imgname].jpg
