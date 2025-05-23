from ultralytics import YOLO

model = YOLO(r'E:\supreet\personal prjt\ML prjts\Football_analysis\model\best.pt')

results = model.predict('E:\supreet\personal prjt\ML prjts\Football_analysis\input_data.mp4', save=True, project='E:\supreet\personal prjt\ML prjts\Football_analysis')

print(results[0])

print('=========================================================')

for box in results[0].boxes:
    print(box)
