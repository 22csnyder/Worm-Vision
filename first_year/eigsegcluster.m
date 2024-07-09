
M=[0, .9, .9,-.7, .7, .7,-.1, .1, .1;
  .9,  0, .9, .7,-.7, .7, .1,-.1, .1;
  .9, .9,  0, .7, .7,-.7, .1, .1,-.1;
 -.7, .7, .7,  0, .8, .8,-.4, .4, .4;
  .7,-.7, .7, .8,  0, .8, .4,-.4, .4;
  .7, .7,-.7, .8, .8,  0, .4, .4,-.4;
 -.1, .1, .1,-.4, .4, .4,  0, .7, .7;
  .1,-.1, .1, .4,-.4, .4, .7,  0, .7;
  .1, .1,-.1, .4, .4,-.4, .7, .7,  0];

M2=[0, .85, .98,-.7,.71, .7,-.1, .09, .11;
    0,  0, .88, .7,-.7, .7, .2, -.1, .12;
    0,  0,   0, .7  .7, -.7, .1, .1, -.1;
    0,  0,   0,  0, .8, .8, -.4, .43, .4;
    0,  0,   0,  0,  0, .8, .4,  -.4, .4;
    0,  0,   0,  0,  0,  0, .37, .44,  -.4;
    0,  0,   0,  0,  0,  0,  0, .69, .72;
    0,  0,   0,  0,  0,  0,  0,  0, .65;
    0,  0,   0,  0,  0,  0,  0,  0,  0];
% M2(M2<=0)=0;

M4=[0, 15, 15,-.7, .71, .7,-.1, .09, .11;
    0,  0, 15, .7, -.7, .7, .2,-.1, .12;
    0,  0,   0, .7, .7,-.7, .1, .1,-.1;
    0,  0,   0,  0, .8, .8,-.4, .43, .4;
    0,  0,   0,  0,  0, .8, .4,-.4, .4;
    0,  0,   0,  0,  0,  0, .37, .44,-.4;
    0,  0,   0,  0,  0,  0,  0, .69, .72;
    0,  0,   0,  0,  0,  0,  0,  0, .65;
    0,  0,   0,  0,  0,  0,  0,  0,  0];

M2=M2+M2';
M4=M4+M4';

M3=M;
M3(M<0)=0;

B =[0,  0,   0,  1,  0,  0,  1,  0,  0;
    0,  0,   0,  0,  1,  0,  0,  1,  0;
    0,  0,   0,  0,  0,  1,  0,  0,  1;
    1,  0,   0,  0,  0,  0,  1,  0,  0;
    0,  1,   0,  0,  0,  0,  0,  1,  0;
    0,  0,   1,  0,  0,  0,  0,  0,  1;
    1,  0,   0,  1,  0,  0,  0,  0,  0;
    0,  1,   0,  0,  1,  0,  0,  0,  0;
    0,  0,   1,  0,  0,  1,  0,  0,  0];

B1=[1,  0,   0,  1,  0,  0,  1,  0,  0;
    0,  1,   0,  0,  1,  0,  0,  1,  0;
    0,  0,   1,  0,  0,  1,  0,  0,  1;
    1,  0,   0,  1,  0,  0,  1,  0,  0;
    0,  1,   0,  0,  1,  0,  0,  1,  0;
    0,  0,   1,  0,  0,  1,  0,  0,  1;
    1,  0,   0,  1,  0,  0,  1,  0,  0;
    0,  1,   0,  0,  1,  0,  0,  1,  0;
    0,  0,   1,  0,  0,  1,  0,  0,  1];

[v2,d2]=eig(M3,B);
[v3,d3]=eig(M3,B);

% M=M4;
% W=M2;
W=M4;

D=eye(9)*diag((W)*ones(9,1));
% (D^-.5)
% W=D\M;
% W=M;



% D=eye(9)*diag((B\M2)*ones(9,1));

N=(D^-.5)*W*(D^-.5);


[vn,an]=eig(N);

[v,a]=eig(M2,B1);
[b,s]=eig(B\M2);

[n,nn]=eig(N,B);%Now this works best--replaced zeros with negatives
[n1,nn1]=eig(N,B1);%Best for M2,M4
clc
[vn(:,1)/norm(vn(:,1)) v(:,1)/norm(v(:,1)) b(:,1)/norm(b(:,1)) n(:,1)/norm(n(:,1)) n1(:,1)/norm(n1(:,1))]


%%
clc
A=[0,1,1;1,0,1;1,1,0];
inv(A)
%%
T=[0 1 1 0 0 0;
   1 0 1 0 0 0;
   1 1 0 0 0 0;
   0 0 0 0 1 1;
   0 0 0 1 0 1;
   0 0 0 1 1 0];

%%
Y=[0 11 8;
   11 0 3;
    8 3 0];
%%

V=[1 0 0 1 0 0 1 0 0;
   0 1 0 0 1 0 0 1 0;
   0 0 1 0 0 1 0 0 1];
