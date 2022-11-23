import cv2

dir1 = "C:/Leia Research/20221018_C2C12 DifferentiationExperiment/20221018_ImmLtf_D4/ImmLtf_D4_A1_Control/ImmLtf_D4_A1_control_01/A1_Overlay_result.tif"
dir2 = "C:/Leia Research/20221018_C2C12 DifferentiationExperiment/20221018_ImmLtf_D4/ImmLtf_D4_A1_Control/ImmLtf_D4_A1_control_01/A1_CH1_result.tif"

ret, img1 = cv2.imreadmulti(dir1)
ret, img2 = cv2.imreadmulti(dir2)

img = img1[0]-img2[0]

cv2.imwrite("C:/Leia Research/20221018_C2C12 DifferentiationExperiment/20221018_ImmLtf_D4/ImmLtf_D4_A1_Control/ImmLtf_D4_A1_control_01/A1_Overlay_result.tif", img)