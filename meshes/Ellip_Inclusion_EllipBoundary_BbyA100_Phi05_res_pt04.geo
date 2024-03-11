//

          
a = 5.0;              
b = 1.0;
Dimratio = 100.0; 
B = Dimratio*b;
A =   Dimratio*a; 
resolution = 0.04;  
scale = 5*Dimratio;   

Point(1) = {0, 0, 0, resolution};
Point(2) = {A, 0, 0, scale*resolution};
//Point(3) = {L, H, 0, scale*resolution};
Point(4) = {0, B, 0, scale*resolution};
Point(5) = {0, b, 0, resolution};
Point(6) = {a, 0, 0, resolution};

//Line(7) = {1, 2};
//Line(8)  = {2,3};
//Line(9)  = {3,4};
//Line(10)  = {4,1};
Ellipse(8) = {2,1,4,4};
Line(19) = {1,6};
Line(20) = {6,2};
Line(21) = {4,5};
Line(22) = {5,1};

Ellipse(11) = {5,1,6,6};

//Curve Loop(12) = {19, 20, 8, 9,21,22}; //whole rectangle
//Curve Loop(23) = { 20, 8, 9,21,11}; //matrix
Curve Loop(23) = { 20, 8,21,11}; //matrix
Curve Loop(24) = { 19, -11, 22}; //inclusion
//

//Plane Surface(13) = {12};
//Line{11} In Surface{13};
//Plane Surface(13) = {12};
Plane Surface(25) = {23}; //matrix
Plane Surface(26) = {24}; //inclusion

Physical Curve("bottom", 14) = {19,20};
//Physical Curve("right", 15) = {8};
//Physical Curve("top", 16) = {9};
Physical Curve("outer", 15) = {8};
Physical Curve("left", 17) = {21,22};
Physical Curve("inclusion_curve", 18) = {11};

Physical Surface("Matrix", 27) = {25};
Physical Surface("Inclusion", 28) = {26};
