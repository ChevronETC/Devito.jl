void doubler(float *a, int nx, int ny)
{
    for (int ix = 0; ix < nx; ix++){
        for (int iy = 0; iy< ny; iy++){
            a[ix*ny+iy] = 2 * a[ix*ny+iy];
        }
    }
}
