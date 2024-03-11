//+
SetFactory("OpenCASCADE");


h = 3;
l =3.1;

Box(1) = {-l,-l,-l, 2*l,2*l,2*l};
Box(2) = {-h,-h,-h, 2*h,2*h,2*h};
// We apply a boolean difference to create the "cube minus one eigth" shape:
BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

Physical Volume("body",4) = {3}; 
