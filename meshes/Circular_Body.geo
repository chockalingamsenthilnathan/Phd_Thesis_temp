//

//L = 5.0;             
//H = 5.0;             
a = 1.0;              
b = 1.0; 
resolution = 0.05;  
//scale = 20.0;   

Point(1) = {0, 0, 0, resolution};
//Point(2) = {L, 0, 0, scale*resolution};
//Point(3) = {L, H, 0, scale*resolution};
//Point(4) = {0, H, 0, scale*resolution};
Point(5) = {0, b, 0, resolution};
Point(6) = {a, 0, 0, resolution};

//Line(7) = {1, 2};
//Line(8)  = {2,3};
//Line(9)  = {3,4};
//Line(10)  = {4,1};
Line(19) = {1,6};
//Line(20) = {6,2};
//Line(21) = {4,5};
Line(22) = {5,1};

Ellipse(11) = {5,1,6,6};

//Curve Loop(12) = {19, 20, 8, 9,21,22}; //whole rectangle
//Curve Loop(23) = { 20, 8, 9,21,11}; //matrix
Curve Loop(24) = { 19, -11, 22}; //inclusion
//

//Plane Surface(13) = {12};
//Line{11} In Surface{13};
//Plane Surface(13) = {12};
//Plane Surface(25) = {23}; //matrix
Plane Surface(26) = {24}; //inclusion

//Physical Curve('bottom', 14) = {7};
Physical Curve('bottom', 14) = {19};
//Physical Curve('right', 15) = {8};
//Physical Curve('top', 16) = {9};
//Physical Curve('left', 17) = {10};
Physical Curve('left', 17) = {22};
Physical Curve("boundary_curve", 18) = {11};

//Physical Surface("Matrix", 27) = {25};
//Physical Surface("Inclusion", 28) = {26};
Physical Surface("Body", 28) = {26};
