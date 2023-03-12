KIMORE Dataset Annotations:
- Exercises are zero-indexed (0,4) while Kimore dataset Exercises were one-indexed (1-5).
- Annotations for each exercise given in separate folders, with ordered naming scheme. Please use video_person_exercise_mapping.csv to find corresponding sample in original KIMORE dataset using subject ID and exercise number.
- Contains Start and End timestamp of each repetition for each data sample in the dataset, created using RBG videos.
- Frame rate of each video sample is 25 which can be used to convert to frame numbers.

kimore_annotations_all_exercises.csv - Combined repetition annotations for all exercises in KIMORE dataset.
video_person_exercise_mapping.csv - Mapping from Kimore Dataset original dataset subject ID to corresponding Video number for each exercise in the annotations.

raw_annotations:
- ass_files - Output annotations from Aegisub Subtitling Software as .ass files for all videos of each exercise
- csv_files - Annotations in .csv files converted from the .ass files
all_annotations.csv - Combined .csv file for repetition timestamps annotations for each video in the exercise. Incomplete repetitions also marked.
all_annotations_with_labels.csv (Only for Exercises 1 and 2 (Kimore 2 and 3)) - Combined .csv file for sub-repetition timestamps annotations for each video in the exercise. Labels (Left, Right) gives direction of subject movement from the front camera perspective.