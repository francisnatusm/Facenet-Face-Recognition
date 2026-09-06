# Face Database (scale to 100–500 people)

Put one folder per person under `database/`. FaceNet will encode every photo and average them into one embedding per person.

## Folder layout

```text
database/
  alice/
    photo1.jpg
    photo2.jpg
  bob/
    bob.png
  charlie/
    front.jpg
    side.jpg
```

You can also use flat files:

```text
database/
  alice.jpg
  bob.png
```

## Tips for better accuracy at 100–500 people

- Prefer **2–5 clear face photos** per person (different angles/lighting)
- Crop faces so the face fills most of the image
- Avoid logos, group photos, or heavily blurred images
- After adding people, run the app and choose **option 4** to rebuild, then save the `.pkl`

## Performance note

Recognition still works with hundreds of people because each person is only a 128-float vector. Building encodings once is the slow part; saving `face_database.pkl` makes later runs fast.
