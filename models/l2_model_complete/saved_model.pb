??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.32unknown8??
?
conv2d_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_40/kernel
}
$conv2d_40/kernel/Read/ReadVariableOpReadVariableOpconv2d_40/kernel*&
_output_shapes
: *
dtype0
t
conv2d_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_40/bias
m
"conv2d_40/bias/Read/ReadVariableOpReadVariableOpconv2d_40/bias*
_output_shapes
: *
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
: *
dtype0
?
conv2d_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_42/kernel
}
$conv2d_42/kernel/Read/ReadVariableOpReadVariableOpconv2d_42/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_42/bias
m
"conv2d_42/bias/Read/ReadVariableOpReadVariableOpconv2d_42/bias*
_output_shapes
:@*
dtype0
{
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_namedense_24/kernel
t
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes
:	?@*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:@*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:@*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
?
Adam/conv2d_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_40/kernel/m
?
+Adam/conv2d_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_40/bias/m
{
)Adam/conv2d_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_41/kernel/m
?
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_41/bias/m
{
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_42/kernel/m
?
+Adam/conv2d_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_42/bias/m
{
)Adam/conv2d_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes
:	?@*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_40/kernel/v
?
+Adam/conv2d_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_40/bias/v
{
)Adam/conv2d_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_40/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_41/kernel/v
?
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_41/bias/v
{
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_42/kernel/v
?
+Adam/conv2d_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_42/bias/v
{
)Adam/conv2d_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_42/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes
:	?@*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?Z
value?ZB?Z B?Z
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
 	keras_api
R
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
h

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
R
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
?
]iter

^beta_1

_beta_2
	`decay
alearning_ratem?m?%m?&m?3m?4m?Em?Fm?Sm?Tm?v?v?%v?&v?3v?4v?Ev?Fv?Sv?Tv?
F
0
1
%2
&3
34
45
E6
F7
S8
T9
 
F
0
1
%2
&3
34
45
E6
F7
S8
T9
?
trainable_variables
regularization_losses
bmetrics

clayers
dlayer_regularization_losses
	variables
enon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
regularization_losses
fmetrics

glayers
hlayer_regularization_losses
	variables
inon_trainable_variables
 
 
 
?
trainable_variables
regularization_losses
jmetrics

klayers
llayer_regularization_losses
	variables
mnon_trainable_variables
 
 
 
?
!trainable_variables
"regularization_losses
nmetrics

olayers
player_regularization_losses
#	variables
qnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
'trainable_variables
(regularization_losses
rmetrics

slayers
tlayer_regularization_losses
)	variables
unon_trainable_variables
 
 
 
?
+trainable_variables
,regularization_losses
vmetrics

wlayers
xlayer_regularization_losses
-	variables
ynon_trainable_variables
 
 
 
?
/trainable_variables
0regularization_losses
zmetrics

{layers
|layer_regularization_losses
1	variables
}non_trainable_variables
\Z
VARIABLE_VALUEconv2d_42/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_42/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
?
5trainable_variables
6regularization_losses
~metrics

layers
 ?layer_regularization_losses
7	variables
?non_trainable_variables
 
 
 
?
9trainable_variables
:regularization_losses
?metrics
?layers
 ?layer_regularization_losses
;	variables
?non_trainable_variables
 
 
 
?
=trainable_variables
>regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
 
 
 
?
Atrainable_variables
Bregularization_losses
?metrics
?layers
 ?layer_regularization_losses
C	variables
?non_trainable_variables
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
?
Gtrainable_variables
Hregularization_losses
?metrics
?layers
 ?layer_regularization_losses
I	variables
?non_trainable_variables
 
 
 
?
Ktrainable_variables
Lregularization_losses
?metrics
?layers
 ?layer_regularization_losses
M	variables
?non_trainable_variables
 
 
 
?
Otrainable_variables
Pregularization_losses
?metrics
?layers
 ?layer_regularization_losses
Q	variables
?non_trainable_variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
?
Utrainable_variables
Vregularization_losses
?metrics
?layers
 ?layer_regularization_losses
W	variables
?non_trainable_variables
 
 
 
?
Ytrainable_variables
Zregularization_losses
?metrics
?layers
 ?layer_regularization_losses
[	variables
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
n
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


?total

?count
?
_fn_kwargs
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?
thresholds
?true_positives
?true_negatives
?false_positives
?false_negatives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?
thresholds
?true_positives
?false_positives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?
thresholds
?true_positives
?false_negatives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

?0
?1
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
?0
?1
?2
?3
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
 
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
?1
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
 
 
 

?0
?1
 
 
 
 
?0
?1
?2
?3
 
 
 

?0
?1
 
 
 

?0
?1
}
VARIABLE_VALUEAdam/conv2d_40/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_40/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_40/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_42/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_42/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_40_inputPlaceholder*/
_output_shapes
:?????????22*
dtype0*$
shape:?????????22
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_40_inputconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_2682725
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_40/kernel/Read/ReadVariableOp"conv2d_40/bias/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp$conv2d_42/kernel/Read/ReadVariableOp"conv2d_42/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Adam/conv2d_40/kernel/m/Read/ReadVariableOp)Adam/conv2d_40/bias/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp+Adam/conv2d_42/kernel/m/Read/ReadVariableOp)Adam/conv2d_42/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp+Adam/conv2d_40/kernel/v/Read/ReadVariableOp)Adam/conv2d_40/bias/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp+Adam/conv2d_42/kernel/v/Read/ReadVariableOp)Adam/conv2d_42/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_2683315
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_40/kernelconv2d_40/biasconv2d_41/kernelconv2d_41/biasconv2d_42/kernelconv2d_42/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_positives_1true_positives_2false_negatives_1Adam/conv2d_40/kernel/mAdam/conv2d_40/bias/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/mAdam/conv2d_42/kernel/mAdam/conv2d_42/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/conv2d_40/kernel/vAdam/conv2d_40/bias/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/vAdam/conv2d_42/kernel/vAdam/conv2d_42/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/v*9
Tin2
02.*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_2683462??
?
i
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2682231

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_2683133?
;conv2d_42_kernel_regularizer_square_readvariableop_resource
identity??2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_42_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
IdentityIdentity$conv2d_42/kernel/Regularizer/add:z:03^conv2d_42/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp
?
N
2__inference_max_pooling2d_42_layer_call_fn_2682237

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_26822312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
*__inference_dense_24_layer_call_fn_2683022

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_26823242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_66_layer_call_and_return_conditional_losses_2682973

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????		@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		@:& "
 
_user_specified_nameinputs
?
?
F__inference_conv2d_40_layer_call_and_return_conditional_losses_2682137

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_40/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2682725
conv2d_40_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_26821172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_40_input
?
?
__inference_loss_fn_3_2683146>
:dense_24_kernel_regularizer_square_readvariableop_resource
identity??1dense_24/kernel/Regularizer/Square/ReadVariableOp?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_24_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentity#dense_24/kernel/Regularizer/add:z:02^dense_24/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp
?
f
J__inference_activation_66_layer_call_and_return_conditional_losses_2682283

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????		@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		@:& "
 
_user_specified_nameinputs
?
K
/__inference_activation_66_layer_call_fn_2682978

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_66_layer_call_and_return_conditional_losses_26822832
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		@:& "
 
_user_specified_nameinputs
?
K
/__inference_activation_64_layer_call_fn_2682942

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_64_layer_call_and_return_conditional_losses_26822492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????00 :& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_41_layer_call_fn_2682185

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_26821772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_65_layer_call_and_return_conditional_losses_2682955

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683052

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
c
G__inference_flatten_12_layer_call_and_return_conditional_losses_2682984

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_65_layer_call_and_return_conditional_losses_2682266

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:????????? 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
?
F__inference_conv2d_41_layer_call_and_return_conditional_losses_2682177

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_41/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
K
/__inference_activation_67_layer_call_fn_2683032

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_67_layer_call_and_return_conditional_losses_26823412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2682191

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
"
cond_false_2683247
identityz
ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_329914e6814341649982eb02de685cc1/part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683057

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
/__inference_sequential_12_layer_call_fn_2682909

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_26825802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_67_layer_call_and_return_conditional_losses_2682341

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_41_layer_call_fn_2682197

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_26821912
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
??
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682817

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?1dense_24/kernel/Regularizer/Square/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_40/BiasAdd?
activation_64/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
activation_64/Relu?
max_pooling2d_40/MaxPoolMaxPool activation_64/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_41/BiasAdd?
activation_65/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
activation_65/Relu?
max_pooling2d_41/MaxPoolMaxPool activation_65/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPool?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2D!max_pooling2d_41/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2
conv2d_42/BiasAdd?
activation_66/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2
activation_66/Relu?
max_pooling2d_42/MaxPoolMaxPool activation_66/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPoolu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshape!max_pooling2d_42/MaxPool:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulflatten_12/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/BiasAdd}
activation_67/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
activation_67/Reluw
dropout_16/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_16/dropout/rate?
dropout_16/dropout/ShapeShape activation_67/Relu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
%dropout_16/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%dropout_16/dropout/random_uniform/min?
%dropout_16/dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dropout_16/dropout/random_uniform/max?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
%dropout_16/dropout/random_uniform/subSub.dropout_16/dropout/random_uniform/max:output:0.dropout_16/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2'
%dropout_16/dropout/random_uniform/sub?
%dropout_16/dropout/random_uniform/mulMul8dropout_16/dropout/random_uniform/RandomUniform:output:0)dropout_16/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@2'
%dropout_16/dropout/random_uniform/mul?
!dropout_16/dropout/random_uniformAdd)dropout_16/dropout/random_uniform/mul:z:0.dropout_16/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@2#
!dropout_16/dropout/random_uniformy
dropout_16/dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_16/dropout/sub/x?
dropout_16/dropout/subSub!dropout_16/dropout/sub/x:output:0 dropout_16/dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout_16/dropout/sub?
dropout_16/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_16/dropout/truediv/x?
dropout_16/dropout/truedivRealDiv%dropout_16/dropout/truediv/x:output:0dropout_16/dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout_16/dropout/truediv?
dropout_16/dropout/GreaterEqualGreaterEqual%dropout_16/dropout/random_uniform:z:0 dropout_16/dropout/rate:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/mulMul activation_67/Relu:activations:0dropout_16/dropout/truediv:z:0*
T0*'
_output_shapes
:?????????@2
dropout_16/dropout/mul?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_16/dropout/Cast?
dropout_16/dropout/mul_1Muldropout_16/dropout/mul:z:0dropout_16/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_16/dropout/mul_1?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMuldropout_16/dropout/mul_1:z:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAdd?
activation_68/SigmoidSigmoiddense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_68/Sigmoid?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource ^conv2d_40/Conv2D/ReadVariableOp*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource ^conv2d_41/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource ^conv2d_42/Conv2D/ReadVariableOp*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource^dense_24/MatMul/ReadVariableOp*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentityactivation_68/Sigmoid:y:0!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_40_layer_call_fn_2682157

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4????????????????????????????????????**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_26821512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_68_layer_call_and_return_conditional_losses_2682414

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_2683120?
;conv2d_41_kernel_regularizer_square_readvariableop_resource
identity??2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_41_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
IdentityIdentity$conv2d_41/kernel/Regularizer/add:z:03^conv2d_41/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp
?p
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682516
conv2d_40_input,
(conv2d_40_statefulpartitionedcall_args_1,
(conv2d_40_statefulpartitionedcall_args_2,
(conv2d_41_statefulpartitionedcall_args_1,
(conv2d_41_statefulpartitionedcall_args_2,
(conv2d_42_statefulpartitionedcall_args_1,
(conv2d_42_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2
identity??!conv2d_40/StatefulPartitionedCall?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?!conv2d_41/StatefulPartitionedCall?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?!conv2d_42/StatefulPartitionedCall?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp? dense_24/StatefulPartitionedCall?1dense_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_input(conv2d_40_statefulpartitionedcall_args_1(conv2d_40_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_40_layer_call_and_return_conditional_losses_26821372#
!conv2d_40/StatefulPartitionedCall?
activation_64/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_64_layer_call_and_return_conditional_losses_26822492
activation_64/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_64/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_26821512"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0(conv2d_41_statefulpartitionedcall_args_1(conv2d_41_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_26821772#
!conv2d_41/StatefulPartitionedCall?
activation_65/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_65_layer_call_and_return_conditional_losses_26822662
activation_65/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_65/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_26821912"
 max_pooling2d_41/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0(conv2d_42_statefulpartitionedcall_args_1(conv2d_42_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_26822172#
!conv2d_42/StatefulPartitionedCall?
activation_66/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_66_layer_call_and_return_conditional_losses_26822832
activation_66/PartitionedCall?
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_66/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_26822312"
 max_pooling2d_42/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_42/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_flatten_12_layer_call_and_return_conditional_losses_26822982
flatten_12/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_26823242"
 dense_24/StatefulPartitionedCall?
activation_67/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_67_layer_call_and_return_conditional_losses_26823412
activation_67/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall&activation_67/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823742
dropout_16/PartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_26823972"
 dense_25/StatefulPartitionedCall?
activation_68/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_68_layer_call_and_return_conditional_losses_26824142
activation_68/PartitionedCall?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_statefulpartitionedcall_args_1"^conv2d_40/StatefulPartitionedCall*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_statefulpartitionedcall_args_1"^conv2d_41/StatefulPartitionedCall*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_statefulpartitionedcall_args_1"^conv2d_42/StatefulPartitionedCall*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_statefulpartitionedcall_args_1!^dense_24/StatefulPartitionedCall*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentity&activation_68/PartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp"^conv2d_41/StatefulPartitionedCall3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp"^conv2d_42/StatefulPartitionedCall3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_40_input
?
e
,__inference_dropout_16_layer_call_fn_2683062

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
/__inference_sequential_12_layer_call_fn_2682924

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_26826562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
!
cond_true_2683246
identityT
ConstConst*
_output_shapes
: *
dtype0*
valueB B.part2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
/__inference_sequential_12_layer_call_fn_2682669
conv2d_40_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_26826562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_40_input
?
f
G__inference_dropout_16_layer_call_and_return_conditional_losses_2682369

inputs
identity?a
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/rateT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape}
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/random_uniform/min}
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/random_uniform/max?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniform?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: 2
dropout/random_uniform/sub?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@2
dropout/random_uniform/mul?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@2
dropout/random_uniformc
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/sub/xq
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: 2
dropout/subk
dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/truediv/x{
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 2
dropout/truediv?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqualp
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????@2
dropout/mul
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/mul_1e
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
/__inference_sequential_12_layer_call_fn_2682593
conv2d_40_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_40_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_26825802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_40_input
?r
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682455
conv2d_40_input,
(conv2d_40_statefulpartitionedcall_args_1,
(conv2d_40_statefulpartitionedcall_args_2,
(conv2d_41_statefulpartitionedcall_args_1,
(conv2d_41_statefulpartitionedcall_args_2,
(conv2d_42_statefulpartitionedcall_args_1,
(conv2d_42_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2
identity??!conv2d_40/StatefulPartitionedCall?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?!conv2d_41/StatefulPartitionedCall?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?!conv2d_42/StatefulPartitionedCall?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp? dense_24/StatefulPartitionedCall?1dense_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallconv2d_40_input(conv2d_40_statefulpartitionedcall_args_1(conv2d_40_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_40_layer_call_and_return_conditional_losses_26821372#
!conv2d_40/StatefulPartitionedCall?
activation_64/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_64_layer_call_and_return_conditional_losses_26822492
activation_64/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_64/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_26821512"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0(conv2d_41_statefulpartitionedcall_args_1(conv2d_41_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_26821772#
!conv2d_41/StatefulPartitionedCall?
activation_65/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_65_layer_call_and_return_conditional_losses_26822662
activation_65/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_65/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_26821912"
 max_pooling2d_41/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0(conv2d_42_statefulpartitionedcall_args_1(conv2d_42_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_26822172#
!conv2d_42/StatefulPartitionedCall?
activation_66/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_66_layer_call_and_return_conditional_losses_26822832
activation_66/PartitionedCall?
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_66/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_26822312"
 max_pooling2d_42/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_42/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_flatten_12_layer_call_and_return_conditional_losses_26822982
flatten_12/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_26823242"
 dense_24/StatefulPartitionedCall?
activation_67/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_67_layer_call_and_return_conditional_losses_26823412
activation_67/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823692$
"dropout_16/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_26823972"
 dense_25/StatefulPartitionedCall?
activation_68/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_68_layer_call_and_return_conditional_losses_26824142
activation_68/PartitionedCall?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_statefulpartitionedcall_args_1"^conv2d_40/StatefulPartitionedCall*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_statefulpartitionedcall_args_1"^conv2d_41/StatefulPartitionedCall*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_statefulpartitionedcall_args_1"^conv2d_42/StatefulPartitionedCall*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_statefulpartitionedcall_args_1!^dense_24/StatefulPartitionedCall*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentity&activation_68/PartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp"^conv2d_41/StatefulPartitionedCall3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp"^conv2d_42/StatefulPartitionedCall3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:/ +
)
_user_specified_nameconv2d_40_input
?
?
E__inference_dense_25_layer_call_and_return_conditional_losses_2682397

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?r
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682580

inputs,
(conv2d_40_statefulpartitionedcall_args_1,
(conv2d_40_statefulpartitionedcall_args_2,
(conv2d_41_statefulpartitionedcall_args_1,
(conv2d_41_statefulpartitionedcall_args_2,
(conv2d_42_statefulpartitionedcall_args_1,
(conv2d_42_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2
identity??!conv2d_40/StatefulPartitionedCall?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?!conv2d_41/StatefulPartitionedCall?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?!conv2d_42/StatefulPartitionedCall?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp? dense_24/StatefulPartitionedCall?1dense_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_40_statefulpartitionedcall_args_1(conv2d_40_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_40_layer_call_and_return_conditional_losses_26821372#
!conv2d_40/StatefulPartitionedCall?
activation_64/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_64_layer_call_and_return_conditional_losses_26822492
activation_64/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_64/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_26821512"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0(conv2d_41_statefulpartitionedcall_args_1(conv2d_41_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_26821772#
!conv2d_41/StatefulPartitionedCall?
activation_65/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_65_layer_call_and_return_conditional_losses_26822662
activation_65/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_65/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_26821912"
 max_pooling2d_41/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0(conv2d_42_statefulpartitionedcall_args_1(conv2d_42_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_26822172#
!conv2d_42/StatefulPartitionedCall?
activation_66/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_66_layer_call_and_return_conditional_losses_26822832
activation_66/PartitionedCall?
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_66/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_26822312"
 max_pooling2d_42/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_42/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_flatten_12_layer_call_and_return_conditional_losses_26822982
flatten_12/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_26823242"
 dense_24/StatefulPartitionedCall?
activation_67/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_67_layer_call_and_return_conditional_losses_26823412
activation_67/PartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823692$
"dropout_16/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_26823972"
 dense_25/StatefulPartitionedCall?
activation_68/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_68_layer_call_and_return_conditional_losses_26824142
activation_68/PartitionedCall?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_statefulpartitionedcall_args_1"^conv2d_40/StatefulPartitionedCall*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_statefulpartitionedcall_args_1"^conv2d_41/StatefulPartitionedCall*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_statefulpartitionedcall_args_1"^conv2d_42/StatefulPartitionedCall*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_statefulpartitionedcall_args_1!^dense_24/StatefulPartitionedCall*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentity&activation_68/PartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp"^conv2d_41/StatefulPartitionedCall3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp"^conv2d_42/StatefulPartitionedCall3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_64_layer_call_and_return_conditional_losses_2682937

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????00 :& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_40_layer_call_fn_2682145

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+??????????????????????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_40_layer_call_and_return_conditional_losses_26821372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?W
?
 __inference__traced_save_2683315
file_prefix/
+savev2_conv2d_40_kernel_read_readvariableop-
)savev2_conv2d_40_bias_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop/
+savev2_conv2d_42_kernel_read_readvariableop-
)savev2_conv2d_42_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_adam_conv2d_40_kernel_m_read_readvariableop4
0savev2_adam_conv2d_40_bias_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableop6
2savev2_adam_conv2d_42_kernel_m_read_readvariableop4
0savev2_adam_conv2d_42_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_40_kernel_v_read_readvariableop4
0savev2_adam_conv2d_40_bias_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableop6
2savev2_adam_conv2d_42_kernel_v_read_readvariableop4
0savev2_adam_conv2d_42_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:0*
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatch?
condStatelessIfStaticRegexFullMatch:output:0"/device:CPU:0*
Tcond0
*	
Tin
 *
Tout
2*
_lower_using_switch_merge(*
_output_shapes
: *%
else_branchR
cond_false_2683247*
output_shapes
: *$
then_branchR
cond_true_26832462
condi
cond/IdentityIdentitycond:output:0"/device:CPU:0*
T0*
_output_shapes
: 2
cond/Identity{

StringJoin
StringJoinfile_prefixcond/Identity:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_40_kernel_read_readvariableop)savev2_conv2d_40_bias_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop+savev2_conv2d_42_kernel_read_readvariableop)savev2_conv2d_42_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_adam_conv2d_40_kernel_m_read_readvariableop0savev2_adam_conv2d_40_bias_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop2savev2_adam_conv2d_42_kernel_m_read_readvariableop0savev2_adam_conv2d_42_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop2savev2_adam_conv2d_40_kernel_v_read_readvariableop0savev2_adam_conv2d_40_bias_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop2savev2_adam_conv2d_42_kernel_v_read_readvariableop0savev2_adam_conv2d_42_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : : @:@:	?@:@:@:: : : : : : : :?:?:?:?::::: : :  : : @:@:	?@:@:@:: : :  : : @:@:	?@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
?
E__inference_dense_24_layer_call_and_return_conditional_losses_2683015

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_24/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
H
,__inference_flatten_12_layer_call_fn_2682989

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_flatten_12_layer_call_and_return_conditional_losses_26822982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2682217

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?2conv2d_42/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
f
J__inference_activation_64_layer_call_and_return_conditional_losses_2682249

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????00 :& "
 
_user_specified_nameinputs
?
f
J__inference_activation_67_layer_call_and_return_conditional_losses_2683027

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?p
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682656

inputs,
(conv2d_40_statefulpartitionedcall_args_1,
(conv2d_40_statefulpartitionedcall_args_2,
(conv2d_41_statefulpartitionedcall_args_1,
(conv2d_41_statefulpartitionedcall_args_2,
(conv2d_42_statefulpartitionedcall_args_1,
(conv2d_42_statefulpartitionedcall_args_2+
'dense_24_statefulpartitionedcall_args_1+
'dense_24_statefulpartitionedcall_args_2+
'dense_25_statefulpartitionedcall_args_1+
'dense_25_statefulpartitionedcall_args_2
identity??!conv2d_40/StatefulPartitionedCall?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?!conv2d_41/StatefulPartitionedCall?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?!conv2d_42/StatefulPartitionedCall?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp? dense_24/StatefulPartitionedCall?1dense_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?
!conv2d_40/StatefulPartitionedCallStatefulPartitionedCallinputs(conv2d_40_statefulpartitionedcall_args_1(conv2d_40_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_40_layer_call_and_return_conditional_losses_26821372#
!conv2d_40/StatefulPartitionedCall?
activation_64/PartitionedCallPartitionedCall*conv2d_40/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????00 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_64_layer_call_and_return_conditional_losses_26822492
activation_64/PartitionedCall?
 max_pooling2d_40/PartitionedCallPartitionedCall&activation_64/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_26821512"
 max_pooling2d_40/PartitionedCall?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0(conv2d_41_statefulpartitionedcall_args_1(conv2d_41_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_41_layer_call_and_return_conditional_losses_26821772#
!conv2d_41/StatefulPartitionedCall?
activation_65/PartitionedCallPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_65_layer_call_and_return_conditional_losses_26822662
activation_65/PartitionedCall?
 max_pooling2d_41/PartitionedCallPartitionedCall&activation_65/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_26821912"
 max_pooling2d_41/PartitionedCall?
!conv2d_42/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0(conv2d_42_statefulpartitionedcall_args_1(conv2d_42_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_26822172#
!conv2d_42/StatefulPartitionedCall?
activation_66/PartitionedCallPartitionedCall*conv2d_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????		@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_66_layer_call_and_return_conditional_losses_26822832
activation_66/PartitionedCall?
 max_pooling2d_42/PartitionedCallPartitionedCall&activation_66/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_26822312"
 max_pooling2d_42/PartitionedCall?
flatten_12/PartitionedCallPartitionedCall)max_pooling2d_42/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_flatten_12_layer_call_and_return_conditional_losses_26822982
flatten_12/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_12/PartitionedCall:output:0'dense_24_statefulpartitionedcall_args_1'dense_24_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_26823242"
 dense_24/StatefulPartitionedCall?
activation_67/PartitionedCallPartitionedCall)dense_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_67_layer_call_and_return_conditional_losses_26823412
activation_67/PartitionedCall?
dropout_16/PartitionedCallPartitionedCall&activation_67/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823742
dropout_16/PartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0'dense_25_statefulpartitionedcall_args_1'dense_25_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_26823972"
 dense_25/StatefulPartitionedCall?
activation_68/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_68_layer_call_and_return_conditional_losses_26824142
activation_68/PartitionedCall?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_statefulpartitionedcall_args_1"^conv2d_40/StatefulPartitionedCall*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_statefulpartitionedcall_args_1"^conv2d_41/StatefulPartitionedCall*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_statefulpartitionedcall_args_1"^conv2d_42/StatefulPartitionedCall*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_statefulpartitionedcall_args_1!^dense_24/StatefulPartitionedCall*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentity&activation_68/PartitionedCall:output:0"^conv2d_40/StatefulPartitionedCall3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp"^conv2d_41/StatefulPartitionedCall3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp"^conv2d_42/StatefulPartitionedCall3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp!^dense_24/StatefulPartitionedCall2^dense_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2F
!conv2d_40/StatefulPartitionedCall!conv2d_40/StatefulPartitionedCall2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_42/StatefulPartitionedCall!conv2d_42/StatefulPartitionedCall2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
E__inference_dense_25_layer_call_and_return_conditional_losses_2683077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
E__inference_dense_24_layer_call_and_return_conditional_losses_2682324

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?1dense_24/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource^MatMul/ReadVariableOp*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_2683107?
;conv2d_40_kernel_regularizer_square_readvariableop_resource
identity??2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_40_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
IdentityIdentity$conv2d_40/kernel/Regularizer/add:z:03^conv2d_40/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp
?
i
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2682151

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?o
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682894

inputs,
(conv2d_40_conv2d_readvariableop_resource-
)conv2d_40_biasadd_readvariableop_resource,
(conv2d_41_conv2d_readvariableop_resource-
)conv2d_41_biasadd_readvariableop_resource,
(conv2d_42_conv2d_readvariableop_resource-
)conv2d_42_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource+
'dense_25_matmul_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identity?? conv2d_40/BiasAdd/ReadVariableOp?conv2d_40/Conv2D/ReadVariableOp?2conv2d_40/kernel/Regularizer/Square/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp?2conv2d_41/kernel/Regularizer/Square/ReadVariableOp? conv2d_42/BiasAdd/ReadVariableOp?conv2d_42/Conv2D/ReadVariableOp?2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?1dense_24/kernel/Regularizer/Square/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?
conv2d_40/Conv2D/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_40/Conv2D/ReadVariableOp?
conv2d_40/Conv2DConv2Dinputs'conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingVALID*
strides
2
conv2d_40/Conv2D?
 conv2d_40/BiasAdd/ReadVariableOpReadVariableOp)conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_40/BiasAdd/ReadVariableOp?
conv2d_40/BiasAddBiasAddconv2d_40/Conv2D:output:0(conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_40/BiasAdd?
activation_64/ReluReluconv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
activation_64/Relu?
max_pooling2d_40/MaxPoolMaxPool activation_64/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_40/MaxPool?
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_41/Conv2D/ReadVariableOp?
conv2d_41/Conv2DConv2D!max_pooling2d_40/MaxPool:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_41/Conv2D?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_41/BiasAdd/ReadVariableOp?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_41/BiasAdd?
activation_65/ReluReluconv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
activation_65/Relu?
max_pooling2d_41/MaxPoolMaxPool activation_65/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_41/MaxPool?
conv2d_42/Conv2D/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_42/Conv2D/ReadVariableOp?
conv2d_42/Conv2DConv2D!max_pooling2d_41/MaxPool:output:0'conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2
conv2d_42/Conv2D?
 conv2d_42/BiasAdd/ReadVariableOpReadVariableOp)conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_42/BiasAdd/ReadVariableOp?
conv2d_42/BiasAddBiasAddconv2d_42/Conv2D:output:0(conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2
conv2d_42/BiasAdd?
activation_66/ReluReluconv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2
activation_66/Relu?
max_pooling2d_42/MaxPoolMaxPool activation_66/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_42/MaxPoolu
flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_12/Const?
flatten_12/ReshapeReshape!max_pooling2d_42/MaxPool:output:0flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_12/Reshape?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulflatten_12/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/BiasAdd}
activation_67/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
activation_67/Relu?
dropout_16/IdentityIdentity activation_67/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_16/Identity?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMuldropout_16/Identity:output:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/BiasAdd?
activation_68/SigmoidSigmoiddense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
activation_68/Sigmoid?
2conv2d_40/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_40_conv2d_readvariableop_resource ^conv2d_40/Conv2D/ReadVariableOp*&
_output_shapes
: *
dtype024
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_40/kernel/Regularizer/SquareSquare:conv2d_40/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_40/kernel/Regularizer/Square?
"conv2d_40/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_40/kernel/Regularizer/Const?
 conv2d_40/kernel/Regularizer/SumSum'conv2d_40/kernel/Regularizer/Square:y:0+conv2d_40/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/Sum?
"conv2d_40/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_40/kernel/Regularizer/mul/x?
 conv2d_40/kernel/Regularizer/mulMul+conv2d_40/kernel/Regularizer/mul/x:output:0)conv2d_40/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/mul?
"conv2d_40/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_40/kernel/Regularizer/add/x?
 conv2d_40/kernel/Regularizer/addAddV2+conv2d_40/kernel/Regularizer/add/x:output:0$conv2d_40/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_40/kernel/Regularizer/add?
2conv2d_41/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource ^conv2d_41/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype024
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_41/kernel/Regularizer/SquareSquare:conv2d_41/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2%
#conv2d_41/kernel/Regularizer/Square?
"conv2d_41/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_41/kernel/Regularizer/Const?
 conv2d_41/kernel/Regularizer/SumSum'conv2d_41/kernel/Regularizer/Square:y:0+conv2d_41/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/Sum?
"conv2d_41/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_41/kernel/Regularizer/mul/x?
 conv2d_41/kernel/Regularizer/mulMul+conv2d_41/kernel/Regularizer/mul/x:output:0)conv2d_41/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/mul?
"conv2d_41/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_41/kernel/Regularizer/add/x?
 conv2d_41/kernel/Regularizer/addAddV2+conv2d_41/kernel/Regularizer/add/x:output:0$conv2d_41/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_41/kernel/Regularizer/add?
2conv2d_42/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_42_conv2d_readvariableop_resource ^conv2d_42/Conv2D/ReadVariableOp*&
_output_shapes
: @*
dtype024
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_42/kernel/Regularizer/SquareSquare:conv2d_42/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2%
#conv2d_42/kernel/Regularizer/Square?
"conv2d_42/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_42/kernel/Regularizer/Const?
 conv2d_42/kernel/Regularizer/SumSum'conv2d_42/kernel/Regularizer/Square:y:0+conv2d_42/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/Sum?
"conv2d_42/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"conv2d_42/kernel/Regularizer/mul/x?
 conv2d_42/kernel/Regularizer/mulMul+conv2d_42/kernel/Regularizer/mul/x:output:0)conv2d_42/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/mul?
"conv2d_42/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_42/kernel/Regularizer/add/x?
 conv2d_42/kernel/Regularizer/addAddV2+conv2d_42/kernel/Regularizer/add/x:output:0$conv2d_42/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_42/kernel/Regularizer/add?
1dense_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource^dense_24/MatMul/ReadVariableOp*
_output_shapes
:	?@*
dtype023
1dense_24/kernel/Regularizer/Square/ReadVariableOp?
"dense_24/kernel/Regularizer/SquareSquare9dense_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2$
"dense_24/kernel/Regularizer/Square?
!dense_24/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_24/kernel/Regularizer/Const?
dense_24/kernel/Regularizer/SumSum&dense_24/kernel/Regularizer/Square:y:0*dense_24/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/Sum?
!dense_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_24/kernel/Regularizer/mul/x?
dense_24/kernel/Regularizer/mulMul*dense_24/kernel/Regularizer/mul/x:output:0(dense_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/mul?
!dense_24/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_24/kernel/Regularizer/add/x?
dense_24/kernel/Regularizer/addAddV2*dense_24/kernel/Regularizer/add/x:output:0#dense_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_24/kernel/Regularizer/add?
IdentityIdentityactivation_68/Sigmoid:y:0!^conv2d_40/BiasAdd/ReadVariableOp ^conv2d_40/Conv2D/ReadVariableOp3^conv2d_40/kernel/Regularizer/Square/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp3^conv2d_41/kernel/Regularizer/Square/ReadVariableOp!^conv2d_42/BiasAdd/ReadVariableOp ^conv2d_42/Conv2D/ReadVariableOp3^conv2d_42/kernel/Regularizer/Square/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp2^dense_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2D
 conv2d_40/BiasAdd/ReadVariableOp conv2d_40/BiasAdd/ReadVariableOp2B
conv2d_40/Conv2D/ReadVariableOpconv2d_40/Conv2D/ReadVariableOp2h
2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2conv2d_40/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2h
2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2conv2d_41/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_42/BiasAdd/ReadVariableOp conv2d_42/BiasAdd/ReadVariableOp2B
conv2d_42/Conv2D/ReadVariableOpconv2d_42/Conv2D/ReadVariableOp2h
2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2conv2d_42/kernel/Regularizer/Square/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2f
1dense_24/kernel/Regularizer/Square/ReadVariableOp1dense_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_2683462
file_prefix%
!assignvariableop_conv2d_40_kernel%
!assignvariableop_1_conv2d_40_bias'
#assignvariableop_2_conv2d_41_kernel%
!assignvariableop_3_conv2d_41_bias'
#assignvariableop_4_conv2d_42_kernel%
!assignvariableop_5_conv2d_42_bias&
"assignvariableop_6_dense_24_kernel$
 assignvariableop_7_dense_24_bias&
"assignvariableop_8_dense_25_kernel$
 assignvariableop_9_dense_25_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count&
"assignvariableop_17_true_positives&
"assignvariableop_18_true_negatives'
#assignvariableop_19_false_positives'
#assignvariableop_20_false_negatives(
$assignvariableop_21_true_positives_1)
%assignvariableop_22_false_positives_1(
$assignvariableop_23_true_positives_2)
%assignvariableop_24_false_negatives_1/
+assignvariableop_25_adam_conv2d_40_kernel_m-
)assignvariableop_26_adam_conv2d_40_bias_m/
+assignvariableop_27_adam_conv2d_41_kernel_m-
)assignvariableop_28_adam_conv2d_41_bias_m/
+assignvariableop_29_adam_conv2d_42_kernel_m-
)assignvariableop_30_adam_conv2d_42_bias_m.
*assignvariableop_31_adam_dense_24_kernel_m,
(assignvariableop_32_adam_dense_24_bias_m.
*assignvariableop_33_adam_dense_25_kernel_m,
(assignvariableop_34_adam_dense_25_bias_m/
+assignvariableop_35_adam_conv2d_40_kernel_v-
)assignvariableop_36_adam_conv2d_40_bias_v/
+assignvariableop_37_adam_conv2d_41_kernel_v-
)assignvariableop_38_adam_conv2d_41_bias_v/
+assignvariableop_39_adam_conv2d_42_kernel_v-
)assignvariableop_40_adam_conv2d_42_bias_v.
*assignvariableop_41_adam_dense_24_kernel_v,
(assignvariableop_42_adam_dense_24_bias_v.
*assignvariableop_43_adam_dense_25_kernel_v,
(assignvariableop_44_adam_dense_25_bias_v
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*?
value?B?-B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_40_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_40_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_41_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_41_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_42_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_42_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_24_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_24_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_25_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_25_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_true_positivesIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_false_positivesIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_false_negativesIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_true_positives_1Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_false_positives_1Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_true_positives_2Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_false_negatives_1Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_40_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_40_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_41_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_41_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_42_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_42_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_24_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_24_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_25_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_25_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_40_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_40_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_41_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_41_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_42_kernel_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_42_bias_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_24_kernel_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_24_bias_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_dense_25_kernel_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_dense_25_bias_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?
f
J__inference_activation_68_layer_call_and_return_conditional_losses_2683089

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
+__inference_conv2d_42_layer_call_fn_2682225

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+???????????????????????????@**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_conv2d_42_layer_call_and_return_conditional_losses_26822172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
K
/__inference_activation_68_layer_call_fn_2683094

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_68_layer_call_and_return_conditional_losses_26824142
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
*__inference_dense_25_layer_call_fn_2683084

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_26823972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?K
?	
"__inference__wrapped_model_2682117
conv2d_40_input:
6sequential_12_conv2d_40_conv2d_readvariableop_resource;
7sequential_12_conv2d_40_biasadd_readvariableop_resource:
6sequential_12_conv2d_41_conv2d_readvariableop_resource;
7sequential_12_conv2d_41_biasadd_readvariableop_resource:
6sequential_12_conv2d_42_conv2d_readvariableop_resource;
7sequential_12_conv2d_42_biasadd_readvariableop_resource9
5sequential_12_dense_24_matmul_readvariableop_resource:
6sequential_12_dense_24_biasadd_readvariableop_resource9
5sequential_12_dense_25_matmul_readvariableop_resource:
6sequential_12_dense_25_biasadd_readvariableop_resource
identity??.sequential_12/conv2d_40/BiasAdd/ReadVariableOp?-sequential_12/conv2d_40/Conv2D/ReadVariableOp?.sequential_12/conv2d_41/BiasAdd/ReadVariableOp?-sequential_12/conv2d_41/Conv2D/ReadVariableOp?.sequential_12/conv2d_42/BiasAdd/ReadVariableOp?-sequential_12/conv2d_42/Conv2D/ReadVariableOp?-sequential_12/dense_24/BiasAdd/ReadVariableOp?,sequential_12/dense_24/MatMul/ReadVariableOp?-sequential_12/dense_25/BiasAdd/ReadVariableOp?,sequential_12/dense_25/MatMul/ReadVariableOp?
-sequential_12/conv2d_40/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_40_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_12/conv2d_40/Conv2D/ReadVariableOp?
sequential_12/conv2d_40/Conv2DConv2Dconv2d_40_input5sequential_12/conv2d_40/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingVALID*
strides
2 
sequential_12/conv2d_40/Conv2D?
.sequential_12/conv2d_40/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_12/conv2d_40/BiasAdd/ReadVariableOp?
sequential_12/conv2d_40/BiasAddBiasAdd'sequential_12/conv2d_40/Conv2D:output:06sequential_12/conv2d_40/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2!
sequential_12/conv2d_40/BiasAdd?
 sequential_12/activation_64/ReluRelu(sequential_12/conv2d_40/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2"
 sequential_12/activation_64/Relu?
&sequential_12/max_pooling2d_40/MaxPoolMaxPool.sequential_12/activation_64/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_40/MaxPool?
-sequential_12/conv2d_41/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02/
-sequential_12/conv2d_41/Conv2D/ReadVariableOp?
sequential_12/conv2d_41/Conv2DConv2D/sequential_12/max_pooling2d_40/MaxPool:output:05sequential_12/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
sequential_12/conv2d_41/Conv2D?
.sequential_12/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_12/conv2d_41/BiasAdd/ReadVariableOp?
sequential_12/conv2d_41/BiasAddBiasAdd'sequential_12/conv2d_41/Conv2D:output:06sequential_12/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
sequential_12/conv2d_41/BiasAdd?
 sequential_12/activation_65/ReluRelu(sequential_12/conv2d_41/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2"
 sequential_12/activation_65/Relu?
&sequential_12/max_pooling2d_41/MaxPoolMaxPool.sequential_12/activation_65/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_41/MaxPool?
-sequential_12/conv2d_42/Conv2D/ReadVariableOpReadVariableOp6sequential_12_conv2d_42_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_12/conv2d_42/Conv2D/ReadVariableOp?
sequential_12/conv2d_42/Conv2DConv2D/sequential_12/max_pooling2d_41/MaxPool:output:05sequential_12/conv2d_42/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2 
sequential_12/conv2d_42/Conv2D?
.sequential_12/conv2d_42/BiasAdd/ReadVariableOpReadVariableOp7sequential_12_conv2d_42_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_12/conv2d_42/BiasAdd/ReadVariableOp?
sequential_12/conv2d_42/BiasAddBiasAdd'sequential_12/conv2d_42/Conv2D:output:06sequential_12/conv2d_42/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2!
sequential_12/conv2d_42/BiasAdd?
 sequential_12/activation_66/ReluRelu(sequential_12/conv2d_42/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2"
 sequential_12/activation_66/Relu?
&sequential_12/max_pooling2d_42/MaxPoolMaxPool.sequential_12/activation_66/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2(
&sequential_12/max_pooling2d_42/MaxPool?
sequential_12/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
sequential_12/flatten_12/Const?
 sequential_12/flatten_12/ReshapeReshape/sequential_12/max_pooling2d_42/MaxPool:output:0'sequential_12/flatten_12/Const:output:0*
T0*(
_output_shapes
:??????????2"
 sequential_12/flatten_12/Reshape?
,sequential_12/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_24_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02.
,sequential_12/dense_24/MatMul/ReadVariableOp?
sequential_12/dense_24/MatMulMatMul)sequential_12/flatten_12/Reshape:output:04sequential_12/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential_12/dense_24/MatMul?
-sequential_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_12/dense_24/BiasAdd/ReadVariableOp?
sequential_12/dense_24/BiasAddBiasAdd'sequential_12/dense_24/MatMul:product:05sequential_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2 
sequential_12/dense_24/BiasAdd?
 sequential_12/activation_67/ReluRelu'sequential_12/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2"
 sequential_12/activation_67/Relu?
!sequential_12/dropout_16/IdentityIdentity.sequential_12/activation_67/Relu:activations:0*
T0*'
_output_shapes
:?????????@2#
!sequential_12/dropout_16/Identity?
,sequential_12/dense_25/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_25_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_12/dense_25/MatMul/ReadVariableOp?
sequential_12/dense_25/MatMulMatMul*sequential_12/dropout_16/Identity:output:04sequential_12/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_12/dense_25/MatMul?
-sequential_12/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_25/BiasAdd/ReadVariableOp?
sequential_12/dense_25/BiasAddBiasAdd'sequential_12/dense_25/MatMul:product:05sequential_12/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_12/dense_25/BiasAdd?
#sequential_12/activation_68/SigmoidSigmoid'sequential_12/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2%
#sequential_12/activation_68/Sigmoid?
IdentityIdentity'sequential_12/activation_68/Sigmoid:y:0/^sequential_12/conv2d_40/BiasAdd/ReadVariableOp.^sequential_12/conv2d_40/Conv2D/ReadVariableOp/^sequential_12/conv2d_41/BiasAdd/ReadVariableOp.^sequential_12/conv2d_41/Conv2D/ReadVariableOp/^sequential_12/conv2d_42/BiasAdd/ReadVariableOp.^sequential_12/conv2d_42/Conv2D/ReadVariableOp.^sequential_12/dense_24/BiasAdd/ReadVariableOp-^sequential_12/dense_24/MatMul/ReadVariableOp.^sequential_12/dense_25/BiasAdd/ReadVariableOp-^sequential_12/dense_25/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????22::::::::::2`
.sequential_12/conv2d_40/BiasAdd/ReadVariableOp.sequential_12/conv2d_40/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_40/Conv2D/ReadVariableOp-sequential_12/conv2d_40/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_41/BiasAdd/ReadVariableOp.sequential_12/conv2d_41/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_41/Conv2D/ReadVariableOp-sequential_12/conv2d_41/Conv2D/ReadVariableOp2`
.sequential_12/conv2d_42/BiasAdd/ReadVariableOp.sequential_12/conv2d_42/BiasAdd/ReadVariableOp2^
-sequential_12/conv2d_42/Conv2D/ReadVariableOp-sequential_12/conv2d_42/Conv2D/ReadVariableOp2^
-sequential_12/dense_24/BiasAdd/ReadVariableOp-sequential_12/dense_24/BiasAdd/ReadVariableOp2\
,sequential_12/dense_24/MatMul/ReadVariableOp,sequential_12/dense_24/MatMul/ReadVariableOp2^
-sequential_12/dense_25/BiasAdd/ReadVariableOp-sequential_12/dense_25/BiasAdd/ReadVariableOp2\
,sequential_12/dense_25/MatMul/ReadVariableOp,sequential_12/dense_25/MatMul/ReadVariableOp:/ +
)
_user_specified_nameconv2d_40_input
?
c
G__inference_flatten_12_layer_call_and_return_conditional_losses_2682298

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout_16_layer_call_fn_2683067

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????@**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_16_layer_call_and_return_conditional_losses_26823742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
K
/__inference_activation_65_layer_call_fn_2682960

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:????????? **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_activation_65_layer_call_and_return_conditional_losses_26822662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_16_layer_call_and_return_conditional_losses_2682374

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_40_input@
!serving_default_conv2d_40_input:0?????????22A
activation_680
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?O
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer-13
layer_with_weights-4
layer-14
layer-15
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?K
_tf_keras_sequential?K{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_12", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": [null, 50, 50, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_68", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": [null, 50, 50, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_68", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy", "AUC", "Precision", "Recall"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_40_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 50, 50, 3], "config": {"batch_input_shape": [null, 50, 50, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_40_input"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 50, 50, 3], "config": {"name": "conv2d_40", "trainable": true, "batch_input_shape": [null, 50, 50, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
?
trainable_variables
regularization_losses
	variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
!trainable_variables
"regularization_losses
#	variables
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_40", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_41", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
+trainable_variables
,regularization_losses
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_41", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_42", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
=trainable_variables
>regularization_losses
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_42", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}}
?
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_68", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_68", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
?
]iter

^beta_1

_beta_2
	`decay
alearning_ratem?m?%m?&m?3m?4m?Em?Fm?Sm?Tm?v?v?%v?&v?3v?4v?Ev?Fv?Sv?Tv?"
	optimizer
f
0
1
%2
&3
34
45
E6
F7
S8
T9"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
f
0
1
%2
&3
34
45
E6
F7
S8
T9"
trackable_list_wrapper
?
trainable_variables
regularization_losses
bmetrics

clayers
dlayer_regularization_losses
	variables
enon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_40/kernel
: 2conv2d_40/bias
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses
fmetrics

glayers
hlayer_regularization_losses
	variables
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
regularization_losses
jmetrics

klayers
llayer_regularization_losses
	variables
mnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!trainable_variables
"regularization_losses
nmetrics

olayers
player_regularization_losses
#	variables
qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_41/kernel
: 2conv2d_41/bias
.
%0
&1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'trainable_variables
(regularization_losses
rmetrics

slayers
tlayer_regularization_losses
)	variables
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
,regularization_losses
vmetrics

wlayers
xlayer_regularization_losses
-	variables
ynon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/trainable_variables
0regularization_losses
zmetrics

{layers
|layer_regularization_losses
1	variables
}non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_42/kernel
:@2conv2d_42/bias
.
30
41"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
5trainable_variables
6regularization_losses
~metrics

layers
 ?layer_regularization_losses
7	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9trainable_variables
:regularization_losses
?metrics
?layers
 ?layer_regularization_losses
;	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
=trainable_variables
>regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Atrainable_variables
Bregularization_losses
?metrics
?layers
 ?layer_regularization_losses
C	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?@2dense_24/kernel
:@2dense_24/bias
.
E0
F1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
Gtrainable_variables
Hregularization_losses
?metrics
?layers
 ?layer_regularization_losses
I	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ktrainable_variables
Lregularization_losses
?metrics
?layers
 ?layer_regularization_losses
M	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Otrainable_variables
Pregularization_losses
?metrics
?layers
 ?layer_regularization_losses
Q	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_25/kernel
:2dense_25/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
Utrainable_variables
Vregularization_losses
?metrics
?layers
 ?layer_regularization_losses
W	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ytrainable_variables
Zregularization_losses
?metrics
?layers
 ?layer_regularization_losses
[	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
@
?0
?1
?2
?3"
trackable_list_wrapper
?
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
?$
?
thresholds
?true_positives
?true_negatives
?false_positives
?false_negatives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?"
_tf_keras_layer?!{"class_name": "AUC", "name": "AUC", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "AUC", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
?
?
thresholds
?true_positives
?false_positives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Precision", "name": "Precision", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
?
thresholds
?true_positives
?false_negatives
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Recall", "name": "Recall", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?trainable_variables
?regularization_losses
?metrics
?layers
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
/:- 2Adam/conv2d_40/kernel/m
!: 2Adam/conv2d_40/bias/m
/:-  2Adam/conv2d_41/kernel/m
!: 2Adam/conv2d_41/bias/m
/:- @2Adam/conv2d_42/kernel/m
!:@2Adam/conv2d_42/bias/m
':%	?@2Adam/dense_24/kernel/m
 :@2Adam/dense_24/bias/m
&:$@2Adam/dense_25/kernel/m
 :2Adam/dense_25/bias/m
/:- 2Adam/conv2d_40/kernel/v
!: 2Adam/conv2d_40/bias/v
/:-  2Adam/conv2d_41/kernel/v
!: 2Adam/conv2d_41/bias/v
/:- @2Adam/conv2d_42/kernel/v
!:@2Adam/conv2d_42/bias/v
':%	?@2Adam/dense_24/kernel/v
 :@2Adam/dense_24/bias/v
&:$@2Adam/dense_25/kernel/v
 :2Adam/dense_25/bias/v
?2?
/__inference_sequential_12_layer_call_fn_2682669
/__inference_sequential_12_layer_call_fn_2682909
/__inference_sequential_12_layer_call_fn_2682593
/__inference_sequential_12_layer_call_fn_2682924?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2682117?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_40_input?????????22
?2?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682516
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682817
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682894
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682455?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_40_layer_call_fn_2682145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
F__inference_conv2d_40_layer_call_and_return_conditional_losses_2682137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
/__inference_activation_64_layer_call_fn_2682942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_64_layer_call_and_return_conditional_losses_2682937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_40_layer_call_fn_2682157?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2682151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_41_layer_call_fn_2682185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
F__inference_conv2d_41_layer_call_and_return_conditional_losses_2682177?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
/__inference_activation_65_layer_call_fn_2682960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_65_layer_call_and_return_conditional_losses_2682955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_41_layer_call_fn_2682197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2682191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_42_layer_call_fn_2682225?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2682217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
/__inference_activation_66_layer_call_fn_2682978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_66_layer_call_and_return_conditional_losses_2682973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_42_layer_call_fn_2682237?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2682231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_flatten_12_layer_call_fn_2682989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_12_layer_call_and_return_conditional_losses_2682984?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_24_layer_call_fn_2683022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_24_layer_call_and_return_conditional_losses_2683015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_67_layer_call_fn_2683032?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_67_layer_call_and_return_conditional_losses_2683027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dropout_16_layer_call_fn_2683067
,__inference_dropout_16_layer_call_fn_2683062?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683057
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683052?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dense_25_layer_call_fn_2683084?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_25_layer_call_and_return_conditional_losses_2683077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_activation_68_layer_call_fn_2683094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_activation_68_layer_call_and_return_conditional_losses_2683089?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_2683107?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_2683120?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_2683133?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_2683146?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
<B:
%__inference_signature_wrapper_2682725conv2d_40_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
"__inference__wrapped_model_2682117?
%&34EFST@?=
6?3
1?.
conv2d_40_input?????????22
? "=?:
8
activation_68'?$
activation_68??????????
J__inference_activation_64_layer_call_and_return_conditional_losses_2682937h7?4
-?*
(?%
inputs?????????00 
? "-?*
#? 
0?????????00 
? ?
/__inference_activation_64_layer_call_fn_2682942[7?4
-?*
(?%
inputs?????????00 
? " ??????????00 ?
J__inference_activation_65_layer_call_and_return_conditional_losses_2682955h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
/__inference_activation_65_layer_call_fn_2682960[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
J__inference_activation_66_layer_call_and_return_conditional_losses_2682973h7?4
-?*
(?%
inputs?????????		@
? "-?*
#? 
0?????????		@
? ?
/__inference_activation_66_layer_call_fn_2682978[7?4
-?*
(?%
inputs?????????		@
? " ??????????		@?
J__inference_activation_67_layer_call_and_return_conditional_losses_2683027X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
/__inference_activation_67_layer_call_fn_2683032K/?,
%?"
 ?
inputs?????????@
? "??????????@?
J__inference_activation_68_layer_call_and_return_conditional_losses_2683089X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ~
/__inference_activation_68_layer_call_fn_2683094K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_conv2d_40_layer_call_and_return_conditional_losses_2682137?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_conv2d_40_layer_call_fn_2682145?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
F__inference_conv2d_41_layer_call_and_return_conditional_losses_2682177?%&I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+??????????????????????????? 
? ?
+__inference_conv2d_41_layer_call_fn_2682185?%&I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+??????????????????????????? ?
F__inference_conv2d_42_layer_call_and_return_conditional_losses_2682217?34I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_conv2d_42_layer_call_fn_2682225?34I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
E__inference_dense_24_layer_call_and_return_conditional_losses_2683015]EF0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ~
*__inference_dense_24_layer_call_fn_2683022PEF0?-
&?#
!?
inputs??????????
? "??????????@?
E__inference_dense_25_layer_call_and_return_conditional_losses_2683077\ST/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? }
*__inference_dense_25_layer_call_fn_2683084OST/?,
%?"
 ?
inputs?????????@
? "???????????
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683052\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
G__inference_dropout_16_layer_call_and_return_conditional_losses_2683057\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? 
,__inference_dropout_16_layer_call_fn_2683062O3?0
)?&
 ?
inputs?????????@
p
? "??????????@
,__inference_dropout_16_layer_call_fn_2683067O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
G__inference_flatten_12_layer_call_and_return_conditional_losses_2682984a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
,__inference_flatten_12_layer_call_fn_2682989T7?4
-?*
(?%
inputs?????????@
? "???????????<
__inference_loss_fn_0_2683107?

? 
? "? <
__inference_loss_fn_1_2683120%?

? 
? "? <
__inference_loss_fn_2_26831333?

? 
? "? <
__inference_loss_fn_3_2683146E?

? 
? "? ?
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2682151?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_40_layer_call_fn_2682157?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2682191?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_41_layer_call_fn_2682197?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2682231?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_42_layer_call_fn_2682237?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682455}
%&34EFSTH?E
>?;
1?.
conv2d_40_input?????????22
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682516}
%&34EFSTH?E
>?;
1?.
conv2d_40_input?????????22
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682817t
%&34EFST??<
5?2
(?%
inputs?????????22
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_12_layer_call_and_return_conditional_losses_2682894t
%&34EFST??<
5?2
(?%
inputs?????????22
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_12_layer_call_fn_2682593p
%&34EFSTH?E
>?;
1?.
conv2d_40_input?????????22
p

 
? "???????????
/__inference_sequential_12_layer_call_fn_2682669p
%&34EFSTH?E
>?;
1?.
conv2d_40_input?????????22
p 

 
? "???????????
/__inference_sequential_12_layer_call_fn_2682909g
%&34EFST??<
5?2
(?%
inputs?????????22
p

 
? "???????????
/__inference_sequential_12_layer_call_fn_2682924g
%&34EFST??<
5?2
(?%
inputs?????????22
p 

 
? "???????????
%__inference_signature_wrapper_2682725?
%&34EFSTS?P
? 
I?F
D
conv2d_40_input1?.
conv2d_40_input?????????22"=?:
8
activation_68'?$
activation_68?????????