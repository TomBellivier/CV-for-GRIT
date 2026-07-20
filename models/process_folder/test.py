from ultralytics import YOLO
m = YOLO("models/process_folder/trained_models/best.pt")
print("task =", m.task)                     # doit être 'pose'
r = m.predict(source="models/datasets/all_species/images/test/Afranthidium_schulthessii_F_001.jpg", conf=0.25, verbose=False)[0]
print("boxes:", 0 if r.boxes is None else len(r.boxes))
print("keypoints:", r.keypoints)            # None => le modèle n'a PAS de tête pose