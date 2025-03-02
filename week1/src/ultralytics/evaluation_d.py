from ultralytics import YOLO



model = YOLO("/ghome/c3mcv02/mcv-c5-team1/week1/checkpoints/yolo/yolo11n.pt") 
DATASET_PATH = "/ghome/c3mcv02/mcv-c5-team1/week1/src/ultralytics/data/data.yaml"

results = model.val(data=DATASET_PATH)

# Print specific metrics
print("Class indices with average precision:", results.ap_class_index)
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)


print("F1 score:", results.box.f1)

print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)

print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)

