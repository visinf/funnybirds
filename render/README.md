# Dataset Generation

Our rendering pipeline follows a client-server structure. We use a node.js server with three.js (js/three.js-master: For convenience we cleaned the three.js repository and only kept the files that are relevant for our application) to host a webpage that renders a scene. After the client calls a specific scene, Puppeteer is used to take a screenshot of the webpage resulting in the final image. The client is a simple Python script, that builds the parameter strings for each scene, calls the service, and stores the images.

### Back-end (run the server)
Install dependencies for npm: ```npm install```

Run the node.js server for rendering: ```node server.js```

Output should be: ```Example app listening at http://:::8081```

You can test if the server is properly running in your own browser by pasting the following line in your browser (if the server is running on a different machine than your browser, make sure to forward the port 8081, e.g., ```ssh -L 8081:127.0.0.1:8081 user@machine```):

```
http://localhost:8081/page?render_mode=default&camera_distance=700&camera_pitch=6.28&camera_roll=1.0&light_distance=300&light_pitch=6.0&light_roll=0.0&beak_model=beak04.glb&beak_color=yellow&foot_model=foot01.glb&eye_model=eye02.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing02.glb&wing_color=green&bg_objects=1,2,3&bg_scale_x=20,2,3&bg_scale_y=20,2,100&bg_scale_z=20,2,100&bg_rot_x=20,2,3&bg_rot_y=1,5,100&bg_rot_z=1,2,100&bg_color=red,green,blue&bg_radius=100,150,200&bg_pitch=0,1,2&bg_roll=0.5,1.5,2.5
```

![Preview of rendered FunnyBird](https://github.com/visinf/funnybirds/blob/main/render/funnybirds_render.png)

_You can even turn the scene with a left-click and by moving the cursor._

### Front-end (rendering the dataset)

> :warning: **Warning**: Puppeteer, a package we use to take screenshots of the rendered scenes, stores temporary files in /tmp that start with "puppeteer...". To avoid storing large amounts of data, we automatically delete all the puppeteer files in /tmp. If the default store location on your machine is a different one, you need to adjust the dictionary. Also, if your /tmp folder contains other puppeteer files that you would like to keep, adjust the code accordingly!

To render the dataset, you have to run the above server and call the following lines in that order:

```
python create_dataset.py --mode train --nr_classes 50 --nr_samples_per_class 1000 --root_path /path/to/datasets --create_classes_json --create_dataset_json --render_dataset
```

```
python create_dataset.py --mode train_part_map --nr_classes 50 --nr_samples_per_class 1000 --root_path /path/to/datasets --render_dataset
```

```
python create_dataset.py --mode test --nr_classes 50 --nr_samples_per_class 10 --root_path /path/to/datasets --create_dataset_json --render_dataset
```

```
python create_dataset.py --mode test_part_map --nr_classes 50 --nr_samples_per_class 10 --root_path /path/to/datasets --render_dataset
```

```
python render_interventions.py --mode test --root_path /path/to/datasets
```

Additionally, copy the parts.json file to /path/to/datasets/FunnyBirds. After following all steps, your folder structure should look like this (```ls /path/to/datasets/FunnyBirds```):

```
classes.json  dataset_test.json  dataset_train.json  parts.json  test  test_interventions  test_part_map  train  train_part_map
```
