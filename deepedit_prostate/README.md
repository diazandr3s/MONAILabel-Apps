# DeepEdit for Prostate Segmentation on T2-Weighted MRI

### Model Overview

Interactive MONAI Label App using DeepEdit to label the prostate over single modality 3D T2-weighted MRI

### Inputs

- 3 channels (T2-weighted MRI + foreground points + background points) (foreground and background channels will be empty before click guidance has been added)

### Output

- 1 channel representing the prostate

### Data

The training images belong to the PROSTATEx Challenges training dataset (https://prostatex.grand-challenge.org/). 
The training labels are provided by an external group (https://github.com/rcuocolo/PROSTATEx_masks/). 
Specifically, 174 T2-weighted images were used to train the DeepEdit model: 

ProstateX-0013
ProstateX-0036
ProstateX-0037
ProstateX-0041
ProstateX-0053
ProstateX-0056
ProstateX-0059
ProstateX-0070
ProstateX-0084
ProstateX-0106
ProstateX-0108
ProstateX-0112
ProstateX-0140
ProstateX-0150
ProstateX-0158
ProstateX-0164
ProstateX-0171
ProstateX-0179
ProstateX-0183
ProstateX-0014
ProstateX-0019
ProstateX-0022
ProstateX-0025
ProstateX-0045
ProstateX-0068
ProstateX-0074
ProstateX-0082
ProstateX-0093
ProstateX-0110
ProstateX-0137
ProstateX-0139
ProstateX-0142
ProstateX-0161
ProstateX-0167
ProstateX-0170
ProstateX-0182
ProstateX-0196
ProstateX-0202
ProstateX-0046
ProstateX-0051
ProstateX-0057
ProstateX-0060
ProstateX-0064
ProstateX-0071
ProstateX-0080
ProstateX-0098
ProstateX-0099
ProstateX-0116
ProstateX-0124
ProstateX-0125
ProstateX-0130
ProstateX-0131
ProstateX-0132
ProstateX-0169
ProstateX-0177
ProstateX-0178
ProstateX-0186
ProstateX-0191
ProstateX-0193
ProstateX-0026
ProstateX-0030
ProstateX-0042
ProstateX-0043
ProstateX-0048
ProstateX-0054
ProstateX-0065
ProstateX-0076
ProstateX-0078
ProstateX-0081
ProstateX-0089
ProstateX-0101
ProstateX-0107
ProstateX-0117
ProstateX-0121
ProstateX-0138
ProstateX-0156
ProstateX-0180
ProstateX-0187
ProstateX-0190
ProstateX-0008
ProstateX-0010
ProstateX-0011
ProstateX-0015
ProstateX-0018
ProstateX-0044
ProstateX-0058
ProstateX-0061
ProstateX-0067
ProstateX-0085
ProstateX-0090
ProstateX-0111
ProstateX-0123
ProstateX-0126
ProstateX-0128
ProstateX-0146
ProstateX-0152
ProstateX-0154
ProstateX-0162
ProstateX-0002
ProstateX-0006
ProstateX-0016
ProstateX-0035
ProstateX-0062
ProstateX-0079
ProstateX-0083
ProstateX-0087
ProstateX-0088
ProstateX-0109
ProstateX-0113
ProstateX-0115
ProstateX-0122
ProstateX-0143
ProstateX-0151
ProstateX-0176
ProstateX-0184
ProstateX-0195
ProstateX-0201
ProstateX-0003
ProstateX-0023
ProstateX-0029
ProstateX-0033
ProstateX-0040
ProstateX-0055
ProstateX-0077
ProstateX-0091
ProstateX-0094
ProstateX-0097
ProstateX-0104
ProstateX-0133
ProstateX-0135
ProstateX-0144
ProstateX-0159
ProstateX-0175
ProstateX-0185
ProstateX-0188
ProstateX-0192
ProstateX-0001
ProstateX-0005
ProstateX-0012
ProstateX-0020
ProstateX-0027
ProstateX-0038
ProstateX-0049
ProstateX-0072
ProstateX-0073
ProstateX-0095
ProstateX-0100
ProstateX-0114
ProstateX-0118
ProstateX-0127
ProstateX-0149
ProstateX-0163
ProstateX-0174
ProstateX-0199
ProstateX-0200
ProstateX-0000
ProstateX-0017
ProstateX-0028
ProstateX-0031
ProstateX-0032
ProstateX-0034
ProstateX-0039
ProstateX-0047
ProstateX-0050
ProstateX-0063
ProstateX-0069
ProstateX-0092
ProstateX-0120
ProstateX-0134
ProstateX-0136
ProstateX-0155
ProstateX-0157
ProstateX-0194
ProstateX-0197

20 T2-weighted images were used to validate the DeepEdit model:

ProstateX-0004
ProstateX-0007
ProstateX-0009
ProstateX-0021
ProstateX-0024
ProstateX-0052
ProstateX-0066
ProstateX-0075
ProstateX-0086
ProstateX-0096
ProstateX-0103
ProstateX-0105
ProstateX-0119
ProstateX-0129
ProstateX-0145
ProstateX-0148
ProstateX-0172
ProstateX-0189
ProstateX-0198
ProstateX-0203

### Validation Set Dice for Simulated Interactions

- 0 interactions: 0.9178
- 1 interactions: 0.9219
- 2 interactions: 0.9245
- 5 interactions: 0.9293
- 10 interactions: 0.9344
- 15 interactions: 0.9360

### 3DSlicer

![DeepEdit for prostate](../docs/images/sample-apps/deepedit_prostate.png)

