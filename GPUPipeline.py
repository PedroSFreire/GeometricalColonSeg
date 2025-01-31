import moderngl
import numpy as np
import nibabel as nib
import time

threshold_air = np.int16(-700)


#handle input
print("Please provide file path or name if in the same folder:")
filePath = input()
start_time = time.time()
nii_img = nib.load(filePath)  # Load .nii.gz file
img_x, img_y, img_z = nii_img.shape

img_data = nii_img.get_fdata().astype(np.int32)  # Extract the data
num_voxel = img_x * img_y * img_z



#load shader

with open("Threshold.glsl", "r") as file: threshold_shader = file.read()

ctx = moderngl.create_standalone_context()

#create buffer
start_time_mem = time.time()
ct_texture = ctx.texture3d((img_x, img_y, img_z), 1, img_data.tobytes(), dtype='i4')
print("shader setup: {:.2f}s".format(time.time() - start_time_mem), flush=True)
ct_texture.bind_to_image(0, read=True, write=True)

shader = ctx.compute_shader(threshold_shader)

shader["threshold"] = threshold_air



start_time_exec = time.time()
shader.run(img_x//4 , img_y//4 , img_z//4)
ctx.finish()
print("time of exec: {:.2f}s".format(time.time() - start_time_exec), flush=True)

#test fast invert
with open("FromGroups.glsl", "r") as revfile: reverse_shader_name = revfile.read()
reverse_shader = ctx.compute_shader(reverse_shader_name)
for i in range(500):
    reverse_shader.run(img_x//4 , img_y//4 , img_z//4)
    ctx.finish()


#output
print("time elapsed: {:.2f}s".format(time.time() - start_time), flush=True)
processed_data = np.frombuffer(ct_texture.read(), dtype=np.int32).reshape((img_x , img_y , img_z))


processed_img = nib.Nifti1Image(processed_data, nii_img.affine)

# Save the processed image
output_file = "thresholded_image.nii.gz"
nib.save(processed_img, output_file)
print("total time elapsed: {:.2f}s".format(time.time() - start_time), flush=True)