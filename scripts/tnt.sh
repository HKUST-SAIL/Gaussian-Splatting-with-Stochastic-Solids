ulimit -n 4096
dataset_folder=/media/baowen/data/dataset/tnt_official
scenes=(Barn Caterpillar Ignatius Meetingroom Truck Courthouse)
move_cpus=(0 0 0 0 0 0)
devices=(cuda cuda cuda cuda cuda cuda)


for idx in "${!scenes[@]}"; do
    scene="${scenes[$idx]}"
    device="${devices[$idx]}"
    move="${move_cpus[$idx]}"
    python train.py -s ${dataset_folder}/${scene} -m output/tnt/${scene} -r 2 --use_decoupled_appearance 3 --data_device ${device}
    args=(-m output/tnt/${scene})
    [[ "$move" == "1" ]] && args+=(--move_cpu)
    PYTORCH_ALLOC_CONF=expandable_segments:True python mesh_extract_tetrahedra.py "${args[@]}"
    python eval_tnt/run.py --dataset-dir ${dataset_folder}/${scene} --traj-path ${dataset_folder}/${scene}/${scene}_COLMAP_SfM.log --ply-path output/tnt/${scene}/recon_post.ply --out-dir output/tnt/${scene}/mesh
done