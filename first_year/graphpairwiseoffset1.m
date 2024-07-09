%This file compares the estimated to the actual offsets in the megaplus
%synthesized set using pairwise W0 to Wi estimation.
%3-3-2015

%Actual
Xi=[215-924,215-400,215-316,215-152,215-394,215-452,215-506,215-394,215-220,1734-930];

%estimated
Xhat=[-787,0,-196,0,0,-200,0,0,0,590];

t=linspace(1,10,10);


hold on
plot(t,Xi,'ro')
plot(t,Xhat,'b<')

xlim([0,length(Xi)+1])
legend('actual','estimated');
title('Pairwise (not global) offset estimation')
xlabel('worm number')
ylabel('vertical offset to worm 0')