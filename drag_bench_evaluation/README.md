# How to Evaluate with DragBench

### Step 1: extract dataset
Extract [DragBench](https://github.com/Yujun-Shi/DragDiffusion/releases/download/v0.1.1/DragBench.zip) into the folder "drag_bench_data".
Resulting directory hierarchy should look like the following:

<br>
drag_bench_data<br>
--- animals<br>
------ JH_2023-09-14-1820-16<br>
------ JH_2023-09-14-1821-23<br>
------ JH_2023-09-14-1821-58<br>
------ ...<br>
--- art_work<br>
--- building_city_view<br>
--- ...<br>
--- other_objects<br>
<br>

### Step 2: train LoRA.
Train one LoRA on each image in drag_bench_data.
To do this, simply execute "run_lora_training.py".
Trained LoRAs will be saved in "drag_bench_lora"

### Step 3: run dragging results
To run dragging results of DragDiffusion on images in "drag_bench_data", simply execute "run_drag_diffusion.py".
Results will be saved in "drag_diffusion_res".

### Step 4: evaluate mean distance and similarity.
To evaluate LPIPS score before and after dragging, execute "run_eval_similarity.py"
To evaluate mean distance between target points and the final position of handle points (estimated by DIFT), execute "run_eval_point_matching.py"


# Expand the Dataset
Here we also provided the labeling tool used by us in the file "labeling_tool.py".
Run this file to get the user interface for labeling your images with drag instructions.