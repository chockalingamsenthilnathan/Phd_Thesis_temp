//+
SetFactory("OpenCASCADE");


h = 3;
l =4;
//+
Rectangle(1) = {-l, -l , 0, 2*l, 2*l, 0};
//+
Rectangle(2) = {-h, -h, 0, 2*h, 2*h, 0};

// We apply a boolean difference to  subtract Surface(2) from Surface(1) 
// to create the inclusion in the matrix:
//
Physical Surface(3) = BooleanDifference{Surface{1}; Delete;  }{ Surface{2}; Delete; };
//+
