# DiffGen

For the codes, visit https://github.com/ZeamoxWang/DiffGen/tree/main.

If you have installed the opencv library, you can run this project directly.

If you want to vectorize some image, you need to save your target.png in current directory. Some parameters of Adam can be finetuned in 2d_triangles.cpp line 1151-1159.  The whole optimization length can be set in line 1108, 1126. The background color can be modified in line 1114. Line 1160 can change the frequency of regeneration and the maximum number of regenerated layers. Line 1057 can change the number of bins. If you do not want to optimize the color of the background, uncomment the line 822. 

The loss of the numbers can be changed in 2d_triangle.h, line 35.

For the whole setting of those experiments, please refer to the technical report. If you have any questions, do not hesitate to contact me.
