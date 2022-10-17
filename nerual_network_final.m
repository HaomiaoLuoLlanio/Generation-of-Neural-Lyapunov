clear all;
%% the dynamic system is 
% dotx = -x + y
% doty = -(x + y)*sin(x) - 3*y;
% n = 2;
eta = 0.02; a = 0.001;
% f = [-x + xy; -y];
 w1 = rand(2,2);
 w2 = rand(2,1);
%% z0 = [x,y]
% z0 = [x; y];
% z1 = w1.*z0;
% v1 = w1.*sigmoid(z1);
% z2 = v1.*w2;
% V = z2;


%% time derivative of the Lyapunov function
% grediant V 
% g_V = w2*diag(dsigmoid(w1*z0))*w1;
% d_V = g_V * f;
%% cost function
% L = LR(d_V + eta, a) + LR(-V + eta, a);
%% traing the function
n = 50;
times = 50;
alpha = 0.01; beta = 0.05;
%% initialize the network
x = [];y = [];
L =[]; total_L =[];
time =1;
L(1) = 1; total_L(1) = 1; dimension = 1;
A =[]; A(1) = 1; w2_1 = []; w2_2 = [];


%% training untile the network is converaged
while (abs(A(time))>0.01)
    total_L(time + 1) = 0;
for i = 2: n 
 x(i) = samprand(-5, 5, dimension); y(i) = samprand(-5, 5, dimension);
 z0 = [x(i);y(i)];
 f = [-x(i) + y(i); -(x(i) + y(i))*sin(x(i)) - 3*y(i)];
 z1 = w1*z0;
 v1 = sigmoid(z1);
 z2 = v1'*w2;
 V = z2;
 g_V = w2'*diag(dsigmoid(w1*z0))*w1;
 d_V = g_V * f;
 L(i) = LR(d_V + eta, a) + LR(-V + eta, a);
 total_L(time + 1) = total_L(time + 1) + L(i);
end
A = gradient(total_L);
w2 = w2 - alpha * A(time);
w1 = w1 - alpha * ones(2)*A(time);
A = gradient(total_L);
% w1 = w1 - beta * gradient(total_L);
time = time + 1;
w2_1(time) = w2(1,1);
w2_2(time) = w2(2,1);
end 


%% plot the Lyapunov function
% V(x1,x2) = v1'*w2
% V(x1,x2) = ((w1*z0).^2)'*w2; V(x1,x2) = [(w1_11*x1 + w1_12*y1)^2 (w1_21*x2 +
% w1_22*y2)^2]*[w2_1;w2_2]; V(x1,x2) = w2(1,1)*(w1(1,1)*x + w1(1,2)*y).^2 +  w2(2,1)*(w1(2,1)*x + w1(2,2)*y).^2
[X, Y] = meshgrid(-5:0.1: 5, -5: 0.1: 5);
V_plot = w2(1,1)*(w1(1,1)*X + w1(1,2)*Y).^2 +  w2(2,1)*(w1(2,1)*X + w1(2,2)*Y).^2;
surf(X,Y,V_plot);hold on;
% plot the time derivative of Lyapunov function
% dV/dt = w2'*diag(dsigmoid(w1*z0))*w1 * f dV/dt = [w2_11
% w2_21]*diag(w1_11*x + w1_12*y, w1_21 + w1_22*y)*[w1_11(-x +x*y)+w1_12(-y)
% w1_21(-x + x*y) +w1_22(-y)];
% diag_w1z0 = [w1(1,1)*X + w1(1,2)*Y 0;0 w1(2,1)*X +w1(2,2)*Y];
%F = [-X + X.*Y; -Y];
F1 = -X + Y; F2 = -(X + Y).*sin(Y) - 3*Y;
% w1*F = w1(1,1)*(-X + X.*Y)-w1(1,2)*Y; w1(2,1)*(-X+X.*Y)-w1(2,1)*Y
dV_plot = w2(1,1)*((w1(1,1)*X + w1(1,2)*Y).*(w1(1,1)*(F1)+ w1(1,2)*F2))+w2(2,1)*((w1(2,1)*X +w1(2,2)*Y).*(w1(2,1)*(F1)+w1(2,1)*F2));
surf(X,Y,dV_plot);
xlabel('x');ylabel('y');zlabel('V OR dV/dt');
figure(2);grid on;
subplot(211);contour(X,Y,V_plot,'ShowText','on');xlabel('x');ylabel('y');title('contour plot of V');
subplot(212);contour(X,Y,dV_plot,'ShowText','on');xlabel('x');ylabel('y');title('contour plot of dV/dt');

figure(3); t = linspace(1,time,time);
plot(t, w2_1);hold on;
plot(t, w2_2);
function y = sigmoid(x)
y = x.^2;
end
function y = dsigmoid(x)
y = 2 * x;
end 
function y = LR(x,a)
if x >= 0
   y = x;
else
   y=a * x;
end
end

function x = samprand(lowerbound,upperbound,dimension)
x = [];
interval = abs((lowerbound - upperbound)/2);
for i = 1: length(dimension(:,1))
    for j = 1: length(dimension(1,:))
        x(i,j) = (lowerbound + upperbound)/2 + interval *(2 * rand(1,1) - 1);
    end 
end
end
