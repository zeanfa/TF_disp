import numpy
import easy_elas as eelas

# constants
patch_size = 9
image_name = "snowflake1"
error_threshold = 12
max_disp = 60
conv_feature_maps = 112
match_th = 0.3
sad_threshold = 800
window_size = 3
std_threshold = 5

eelas.disp_map_easyELAS(patch_size, max_disp, match_th, conv_feature_maps, "../samples/" + image_name + "/", image_name)


