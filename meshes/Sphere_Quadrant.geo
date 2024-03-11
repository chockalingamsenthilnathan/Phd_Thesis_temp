//+
rad = DefineNumber[ 1.0, Name "Parameters/rad" ];
//+
ndens = DefineNumber[ 0.1, Name "Parameters/ndens" ];
//+
Point(1) = {0, 0, 0, ndens};
//+
Point(2) = {rad, 0, 0, ndens};
//+
//Point(3) = {-rad, 0, 0, ndens};
//+
Point(4) = {0, rad, 0, ndens};
//+
//Point(5) = {0, -rad, 0, ndens};
//+
Point(6) = {0, 0, rad, ndens};
//+
//Point(7) = {0, 0, -rad, ndens};
//+
Circle(1) = {2, 1, 4};
//+
//Circle(2) = {4, 1, 3};
//+
//Circle(3) = {3, 1, 5};
//+
//Circle(4) = {5, 1, 2};
//+
Circle(5) = {6, 1, 2};
//+
//Circle(6) = {6, 1, 3};
//+
//Circle(7) = {3, 1, 7};
//+
//Circle(8) = {7, 1, 2};
//+
Circle(9) = {4, 1, 6};
//+
//Circle(10) = {6, 1, 5};
//+
//Circle(11) = {5, 1, 7};
//+
//Circle(12) = {7, 1, 4};

Line(13) = {4,1};
Line(14) = {1,2};
Line(15) = {6,1};

Curve Loop(1) = {1,9,5};
Curve Loop(2) = {1,13,14};
Curve Loop(3) = {9,15,-13};
Curve Loop(4) = {5,-14,-15};

Surface(1) = {1};
Surface(2) = {-2};
Surface(3) = {-3};
Surface(4) = {-4};

Surface Loop(1) = {1,2,3,4};

Volume(1) = {1};

Physical Surface("boundary_surface",1) = {1};
Physical Surface("ZZero",2) = {2};
Physical Surface("XZero",3) = {3};
Physical Surface("YZero",4) = {4};

Physical Volume("body",1) = {1}; 
