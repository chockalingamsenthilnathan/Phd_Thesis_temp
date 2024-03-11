//+
SetFactory("OpenCASCADE");


h_ring = 0.5;
xring = 3.0;
//+
Rectangle(1) = {xring, 0 , 0,h_ring,  h_ring, 0};

Physical Surface(2) = {1};
