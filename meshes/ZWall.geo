//+
SetFactory("OpenCASCADE");


b_wall = 15.0;
z_wall =3.0;
t_wall = 0.3;
//+
Rectangle(1) = {0, z_wall , 0,b_wall,  t_wall, 0};

Physical Surface(2) = {1};
